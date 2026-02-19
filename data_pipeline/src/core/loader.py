import fitz
from datetime import datetime, timezone, timedelta
from langchain.schema import Document

from core.util import get_path, get_logger

logger = get_logger(__name__)


class SlideDocLoader():
    def __init__(self, doc_paths, category):
        self.doc_paths = doc_paths
        self.category = category
        
    def __call__(self):
        for fp in self.doc_paths:
            fp = get_path(fp).resolve()
            doc = Document(
                page_content='',
                metadata={
                    'category': self.category,
                    'title': None,
                    'keywords': None,
                    'source_fn': fp.name,
                    'source': fp.name,
                    'embedding': None,
                    'date': datetime.now(),
                    'importance': None,
                    'importance_reason': None,
                    '_local_path': fp,
                }
            )
            yield doc


class BatchSlidePageDocLoader:
    def __init__(self, doc_path, category, page_range):
        self.doc_path = get_path(doc_path)
        self.category = category
        self.start_page, self.end_page = page_range
        
    def __call__(self):
        pdf = fitz.open(self.doc_path)
        
        for page_num in range(self.start_page, self.end_page):
            doc = Document(
                page_content='',
                metadata={
                    'category': self.category,
                    'title': None,
                    'keywords': None,
                    'source_fn': self.doc_path.name,
                    'source': f"{self.doc_path.name}/p{page_num+1}",
                    'embedding': None,
                    'date': datetime.now(),
                    'importance': None,
                    'importance_reason': None,
                    '_local_path': self.doc_path,
                    '_page_num': page_num,
                }
            )
            yield doc
            
        pdf.close()


class BotInputLoader:
    def __init__(self, content, date_str=None):
        self.content = content
        self.parsed_date = self._parse_date(date_str)
        
    def _parse_date(self, date_str):
        if date_str:
            try:
                date_parts = str(date_str).split('-')
                year = int(date_parts[0])
                month = int(date_parts[1]) if len(date_parts) > 1 else 1
                day = int(date_parts[2]) if len(date_parts) > 2 else 1
                return datetime(year, month, day)
            except (ValueError, IndexError):
                logger.warning(f'Invalid date format: {date_str}, using current date')
                return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))
        else:
            return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))
        
    def __call__(self):
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        document = Document(
            page_content = self.content,
            metadata = {
                'category': 'observation',
                'title': None,
                'keywords': None,
                'source_fn': None,
                'source': f"bot/{current_time}",
                'date': self.parsed_date,
                'importance': None,
                'importance_reason': None,
                'embedding': None
            }
        )
        return [document]
    

class WebAppInputLoader:
    def __init__(self, content, date_str, category):
        self.content = content
        self.parsed_date = self._parse_date(date_str)
        self.category = category
        
    def _parse_date(self, date_str):
        if date_str:
            try:
                date_parts = str(date_str).split('-')
                year = int(date_parts[0])
                month = int(date_parts[1]) if len(date_parts) > 1 else 1
                day = int(date_parts[2]) if len(date_parts) > 2 else 1
                return datetime(year, month, day)
            except (ValueError, IndexError):
                logger.warning(f'Invalid date format: {date_str}, using current date')
                return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))
        else:
            return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))
        
    def __call__(self):
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        document = Document(
            page_content = self.content,
            metadata = {
                'category': self.category,
                'title': None,
                'keywords': None,
                'source_fn': None,
                'source': f"bot/{current_time}",
                'date': self.parsed_date,
                'importance': None,
                'importance_reason': None,
                'embedding': None
            }
        )
        return [document]