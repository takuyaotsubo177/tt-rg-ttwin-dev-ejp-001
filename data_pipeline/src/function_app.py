import platform
import uuid
import shutil
import time
import fitz
from datetime import datetime, timezone
from dotenv import load_dotenv
import azure.functions as func
from azure.search.documents.models import QueryType
from unittest import mock

from core.indexer import ExpertTwinIndexer
from core.loader import SlideDocLoader, BatchSlidePageDocLoader, WebAppInputLoader
from core.pipeline import (
    CreateSemanticKeyAgent, TextEmbeddingAgent, CreateIndexAgent, ScoreImportanceAgent,
    DocToImgAgent, DocImgListToTextAgent, OCRWholeSlideByGptAgent, DocImgToTextAgent, OCRSinglePageByGptAgent
)
from core.util import Conf, Client, get_path, get_logger, dump_as
from core.error import SkipRequest
from core.queue_io import QueueIO

APP = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

load_dotenv()
Conf.set('./function_app_conf')

logger = get_logger(__name__)


@APP.blob_trigger(
    arg_name='blob',
    path='knowledge/slide/{name}',
    connection='Storage'
)
def create_index_slide(blob: func.InputStream):
    logger.info(f'*** Start create_index_slide: {blob.name} ***')
    start_time = time.perf_counter()

    proceed_indexing = _should_proceed_indexing(blob.name)
    if not proceed_indexing:
        return
    
    C = Conf.get()
    PC = C.pipeline.ExpertTwinIndexer

    indexer = ExpertTwinIndexer(
        PC.index_name,
        PC.lang,
        PC.semantic_name,
        C.openai.embed.base
    )
    indexer()

    src_dir, dst_dir = _setup_directories()
    whole_content = None
    try:
        doc_path = _cp_blob_to_local(blob.name, src_dir)
        doc_category = 'slide'

        # 同一スライドのインデックスが存在する場合は削除
        _delete_index(
            PC.index_name,
            filter_by=f"category eq '{doc_category}' and source eq '{doc_path.name}'"
        )

        # 1. 全体構成インデックスの作成
        loader, pipelines = _get_slide_whole_pdf_module(
            [str(doc_path)],
            doc_category,
            PC.index_name,
            dst_dir
        )
        
        # 全体構成情報を取得
        for doc in loader():
            is_ok = True
            for p in pipelines:
                try:
                    p(doc)
                except SkipRequest:
                    is_ok = False
                    break

            if is_ok:
                whole_content = doc.page_content
                m = doc.metadata
                logger.info(f"Success create index: title={m['title']}, source={m['source']}, category={m['category']}, importance={m['importance']}")
        
        # 2. ページ単位のインデックス作成をキューに登録
        _queue_slide_page_processing(doc_path, whole_content)
        
        _update_blob_metadata(blob.name)

    except Exception as e:
        raise RuntimeError(f'fail to execute by {blob.name}, reason:{e}')

    finally:
        import gc
        gc.collect()
        if src_dir.exists():
            shutil.rmtree(src_dir)
        if dst_dir.exists():
            shutil.rmtree(dst_dir)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logger.info(f'*** End create_index_slide elapsed(sec): {execution_time} ***')


# -------------------------------------------------- queue trigger
@APP.queue_trigger(
    arg_name='queue',
    queue_name='slide-page-queue',
    connection='Storage'
)
def process_slide_page_queue(queue: func.QueueMessage):
    logger.info(f'*** Start process_slide_page_queue ***')
    start_time = time.perf_counter()

    try:
        data = QueueIO.decode('json', queue)
        data = data[0]
        if not data:
            logger.error('Empty queue message')
            return
            
        doc_path = data.get('doc_path')
        page_range = data.get('page_range')
        whole_content = data.get('whole_content')
        
        if not all([doc_path, page_range]):
            logger.error(f'Invalid queue message: {data}')
            return
        
        C = Conf.get()
        PC = C.pipeline.ExpertTwinIndexer
        index_name = PC.index_name
        doc_category = 'slide_page'
            
        src_dir, dst_dir = _setup_directories()
        try:
            local_doc_path = _cp_blob_to_local(doc_path, src_dir)
            start_page, end_page = page_range

            for page_num in range(start_page, end_page):
                page_path = f"{local_doc_path.name}/p{page_num+1}"
                _delete_index(
                    index_name,
                    filter_by=f"category eq '{doc_category}' and source eq '{page_path}'"
                )
            
            loader, pipelines = _get_slide_page_pdf_module(
                local_doc_path,
                doc_category,
                index_name,
                page_range,
                dst_dir,
                whole_content
            )
            _create_index(loader, pipelines)
            
        finally:
            import gc
            gc.collect()
            if src_dir.exists():
                shutil.rmtree(src_dir)
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
                
    except Exception as e:
        logger.error(f'Failed to process slide page queue: {e}')
        raise RuntimeError(f'Failed to process slide page queue: {e}')
        
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logger.info(f'*** End process_slide_page_queue elapsed(sec): {execution_time} ***')


@APP.queue_trigger(
    arg_name='queue',
    queue_name='create-index-by-webapp',
    connection='Storage'
)
def process_create_index_by_webapp_queue(queue: func.QueueMessage):
    logger.info(f'*** Start process_create_index_by_webapp_queue ***')
    start_time = time.perf_counter()

    try:
        data = QueueIO.decode('json', queue)
        for item in data:
            date_str = item.get('date', '')
            category = item.get('category', '')
            content = item.get('content', '')
            
            if not content:
                logger.error('Content is empty, skipping index creation')
                return
            
            C = Conf.get()
            PC = C.pipeline.ExpertTwinIndexer

            indexer = ExpertTwinIndexer(
                PC.index_name,
                PC.lang,
                PC.semantic_name,
                C.openai.embed.base
            )
            indexer()
            
            # Get loader and pipelines for bot input
            loader, pipelines = _get_webapp_input_module(content, date_str, category, PC.index_name)
            
            # Create index using the standard function
            _create_index(loader, pipelines)
        
    except Exception as e:
        raise RuntimeError(f'Failed to process create index queue: {e}')
        
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logger.info(f'*** End process_create_index_by_webapp_queue elapsed(sec): {execution_time} ***')

# -------------------------------------------------- blob
def _setup_directories():
    uid = str(uuid.uuid4())
    if platform.system() == 'Linux':
        src_dir = get_path(f'/tmp/{uid}/src', mkdirs=True)
        dst_dir = get_path(f'/tmp/{uid}/dst', mkdirs=True)
    else:
        src_dir = get_path(f'./tmp/{uid}/src', mkdirs=True)
        dst_dir = get_path(f'./tmp/{uid}/dst', mkdirs=True)
    return src_dir, dst_dir

def _cp_blob_to_local(frm_blob, to_dir):
    container_name, frm_name = frm_blob.split('/', 1)

    cli = Client.get_az_blob_client()
    raw = cli.get_blob_client(container=container_name, blob=frm_name) \
        .download_blob() \
        .readall()

    to_path = to_dir / frm_name
    to_path.parent.mkdir(parents=True, exist_ok=True)
    with to_path.open('wb') as f:
        f.write(raw)

    return to_path

def _should_proceed_indexing(frm_blob):
    container_name, frm_name = frm_blob.split('/', 1)

    cli = Client.get_az_blob_client()
    con = cli.get_blob_client(container=container_name, blob=frm_name)

    metadata = con.get_blob_properties().metadata or {}
    index_created_at_str = metadata.get('index_created_at')

    if index_created_at_str:
        try:
            index_created_at = datetime.strptime(index_created_at_str, "%Y%m%d")
            logger.info(f'Index already created at {index_created_at.strftime("%Y-%m-%d")}. Skipping indexing.: {frm_name}')
            return False
        except ValueError:
            logger.warning(f'Invalid timestamp format in metadata: {index_created_at_str}. Proceeding with indexing.: {frm_name}')
            return True
    else:
        logger.info(f'No index_created_at metadata found. Proceeding with indexing.: {frm_name}')
        return True

def _update_blob_metadata(frm_blob):
    container_name, frm_name = frm_blob.split('/', 1)

    cli = Client.get_az_blob_client()
    con = cli.get_blob_client(container=container_name, blob=frm_name)

    current_metadata = con.get_blob_properties().metadata or {}
    new_metadata = {'index_created_at': datetime.now(timezone.utc).strftime("%Y%m%d")}
    updated_metadata = {**current_metadata, **new_metadata}

    con.set_blob_metadata(updated_metadata)
    logger.info(f'Metadata updated with {new_metadata}')

def _queue_slide_page_processing(doc_path, whole_content=None):
    pdf = fitz.open(doc_path)
    page_count = pdf.page_count
    pdf.close()
    
    rel_path = f"knowledge/slide/{doc_path.name}"
    batch_size = 10
    queue_io = QueueIO('slide-page-queue')
    
    try:
        batches = []
        for start_page in range(0, page_count, batch_size):
            end_page = min(start_page + batch_size, page_count)
            batch = {
                'doc_path': rel_path,
                'page_range': [start_page, end_page],
                'whole_content': whole_content
            }
            batches.append(batch)
        
        queue_io.write('json', batches, size=1)
        logger.info(f"Queued {len(batches)} batches for {rel_path}, total pages: {page_count}")
        
    finally:
        queue_io.close()


# -------------------------------------------------- modular
def _get_slide_whole_pdf_module(doc_path_ptns, category, index_name, dst_dir):
    loader = SlideDocLoader(doc_path_ptns, category)
    
    pipelines = [
        DocToImgAgent(dst_dir),
        DocImgListToTextAgent(ocr_agent=OCRWholeSlideByGptAgent()),
        CreateSemanticKeyAgent(
            input_keys=['title', 'page_content'],
            output_by={'keywords': 'keywords', 'title': 'title'}
        ),
        TextEmbeddingAgent(input_keys=['title', 'page_content']),
        CreateIndexAgent(
            index_name,
            insert_meta_keys=[
                'category',
                'title',
                'keywords',
                'source_fn',
                'source',
                'date',
                'importance',
                'importance_reason',
                'embedding',
            ]
        ),
    ]

    return loader, pipelines


def _get_slide_page_pdf_module(doc_path, doc_category, index_name, page_range, dst_dir, whole_content=None):   
    loader = BatchSlidePageDocLoader(doc_path, doc_category, page_range)
    pipelines = [
        DocToImgAgent(dst_dir),
        DocImgToTextAgent(
            ocr_agent=OCRSinglePageByGptAgent(),
            cleanup_dir=False
        ),
        CreateSemanticKeyAgent(
            input_keys=['title', 'page_content'],
            output_by={'keywords': 'keywords', 'title': 'title'}
        ),
        TextEmbeddingAgent(input_keys=['title', 'page_content']),
        CreateIndexAgent(
            index_name,
            insert_meta_keys=[
                'category',
                'title',
                'keywords',
                'source_fn',
                'source',
                'date',
                'importance',
                'importance_reason',
                'embedding',
            ]
        ),
    ]
    # loaderにwhole_contentを渡すためのラッパー
    if whole_content:
        orig_loader = loader
        def wrapper():
            for doc in orig_loader():
                doc.metadata['_whole_content'] = whole_content
                yield doc
        loader = wrapper
    return loader, pipelines


def _get_webapp_input_module(content, date_str, category, index_name):
    loader = WebAppInputLoader(content, date_str, category)
    
    pipelines = [
        CreateSemanticKeyAgent(
            input_keys=['page_content'],
            output_by={'keywords': 'keywords', 'title': 'title'}
        ),
    ]
    if category in ('observation', 'peer_impression', 'interview'):
        pipelines.append(
            ScoreImportanceAgent(
                input_keys=['page_content'],
                output_by={'importance': 'importance', 'importance_reason': 'importance_reason'}
            )
        )
    pipelines.extend([
        TextEmbeddingAgent(input_keys=['title', 'page_content']),
        CreateIndexAgent(
            index_name,
            insert_meta_keys=[
                'category',
                'title',
                'keywords',
                'source',
                'source_fn',
                'date',
                'importance',
                'importance_reason',
                'embedding',
            ]
        ),
    ])
    
    return loader, pipelines

# -------------------------------------------------- indexer
def _delete_index(index_name, filter_by):
    assert filter_by, 'filter_by is required'

    client = Client.get_az_search_client(index_name)
    raw = client.search(
        query_type=QueryType.SIMPLE,
        query_language='ja-jp',
        search_text='',
        search_mode='all',
        select=['id'],
        filter=filter_by
    )

    ids = []
    for page in raw.by_page():
        for p in page:
            ids.append({'id': p['id']})

    logger.info(f'delete index where {filter_by}, target_row_cnt:{len(ids)}')
    if ids:
        client.delete_documents(ids)

def _create_index(loader, pipelines):
    for doc in loader():
        is_ok = True
        for p in pipelines:
            try:
                p(doc)
            except SkipRequest:
                is_ok = False
                break

        if is_ok:
            m = doc.metadata
            logger.info(f"Success create index: title={m['title']}, source={m['source']}, category={m['category']}, importance={m['importance']}")

# -------------------------------------------------- test
def _test():
    blob_mock = mock.Mock(spec=func.InputStream)
    blob_mock.name = 'knowledge/slide/sample1.pdf'
    blob_func = create_index_slide.build().get_user_function()
    blob_func(blob_mock)

    sample_queue = [{
        'doc_path': 'knowledge/slide/sample1.pdf',
        'page_range': [0, 2],
        'whole_content': 'Sample whole content for testing'
    }]
    message_content = dump_as("json", sample_queue).encode('utf-8')
    queue_mock = mock.Mock()
    queue_mock.get_body.return_value = message_content
    queue_func = process_slide_page_queue.build().get_user_function()
    queue_func(queue_mock)
    
    sample_queue = [{
        'date': '2024-01-01',
        'content': 'This is a test content from bot interface.',
        'category': 'observation'
    }]
    message_content = dump_as("json", sample_queue).encode('utf-8')
    queue_mock = mock.Mock()
    queue_mock.get_body.return_value = message_content
    queue_func = process_create_index_by_webapp_queue.build().get_user_function()
    queue_func(queue_mock)

@APP.route(route='test')
def test(req: func.HttpRequest) -> func.HttpResponse:
    logger.info(f'*** Start Test ***')
    try:
        _test()
        return func.HttpResponse('OK')
    except Exception as e:
        e = str(e)
        return e


if __name__ == '__main__':
    _test()