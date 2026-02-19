# util.py

import os
import pathlib
import yaml
import json
import re
import pandas as pd
from io import StringIO
from logging import getLogger, StreamHandler, Formatter, INFO
from omegaconf import OmegaConf
import openpyxl
from pptx import Presentation
import neologdn
import pprint
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.storage.blob import BlobServiceClient

# -------------------------------------------------- log
def get_logger(name):
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'))

    logger = getLogger(name)
    logger.setLevel(INFO)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

# -------------------------------------------------- conf
# FIXME:
# - langchain用に、AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT をsetしているが、langchain以外にも使いまわす関数なので、削除し、clientの引数指定に変える
# - getした際に、_cacheがなければ処理を落としていいのでは?

class Conf():
    _cache = {}

    @classmethod
    def set(cls, *names, dump=False):
        raws = []
        for n in names:
            r = OmegaConf.load(f'{n}.yaml')
            raws.append(r)
        conf = OmegaConf.merge(*raws)

        for k in conf.get('env', {}).keys():
            v = os.getenv(k)
            conf.env[k] = v
            assert v, f'NotFound env:{k}'

        OmegaConf.set_readonly(conf, True)
        cls._cache = conf

        # validate
        _conf = OmegaConf.to_container(conf, resolve=True)

        if dump:
            pprint.pprint(_conf)

    @classmethod
    def get(cls):
        if not cls._cache:
            cls.set('base')
        return cls._cache

# -------------------------------------------------- client
class Client():
    _cache = {}

    @classmethod
    def get_az_openai_client(cls):
        key = 'openai'
        cli = cls._cache.get(key)
        if cli:
            return cli

        C = Conf.get()
        cli = AzureOpenAI(
            api_key=C.openai.key,
            api_version=C.openai.version,
            azure_endpoint=C.openai.endpoint,
        )

        cls._cache[key] = cli
        return cli

    @classmethod
    def get_index_client(cls):
        key = 'index'
        cli = cls._cache.get(key)
        if cli:
            return cli

        conf = Conf.get()
        cli = SearchIndexClient(
            credential=AzureKeyCredential(conf.search.key),
            endpoint=conf.search.endpoint
        )
        cls._cache[key] = cli

        return cli

    @classmethod
    def get_az_search_client(cls, index_name):
        cli = cls._cache.get(index_name)
        if cli:
            return cli

        C = Conf.get()
        cli = SearchClient(
            credential=AzureKeyCredential(C.search.key),
            endpoint=C.search.endpoint,
            index_name=index_name
        )

        cls._cache[index_name] = cli
        return cli

    @classmethod
    def get_az_blob_client(cls):
        key = 'blob'
        cli = cls._cache.get(key)
        if cli:
            return cli

        C = Conf.get()
        cli = BlobServiceClient.from_connection_string(C.storage.connect)

        cls._cache[key] = cli
        return cli

    @classmethod
    def close(cls):
        for cli in cls._cache.values():
            if hasattr(cli, 'close'):
                cli.close()
        cls._cache = {}

# -------------------------------------------------- text
def load_as(format, data):
    # structured str to collection
    if data is None:
        return data
    elif type(data) != str:
        return data

    if format == 'yaml':
        return yaml.safe_load(data)
    elif format == 'json':
        return json.loads(data)

    raise ValueError(f'Undefined format:{format}')

def dump_as(format, data):
    # collection to structured str
    if data is None:
        return data
    elif type(data) not in (dict, list):
        return data

    if format == 'json':
        return json.dumps(data, ensure_ascii=False)

    raise ValueError(f'Undefined format:{format}')

NORM_PTN = {
    'neologdn': None,
    'escape.ai_search_fulltext': re.compile(r'([{}])'.format('\+\-\&\|\!\(\)\{\}\[\]\^\"\~\*\?\:\\\/')),
    'rm.code_block': re.compile(r'^\s*```(markdown|json)\n|```\s*$'),
}
def norm(_str, *ptn_keys):
    if not _str:
        return _str

    for k in ptn_keys:
        ptn = NORM_PTN[k]
        if k == 'neologdn':
            _str = neologdn.normalize(_str)
        elif k.startswith('escape.'):
            _str = ptn.sub(r'\\\1', _str)
        elif k.startswith('rm.'):
            _str = ptn.sub('', _str)
        else:
            raise ValueError(f'Unexpeted ptn: {k}')

        return _str


# -------------------------------------------------- delete
# FIXME:
# - bc_sales/_toy/file_io.pyを、core/ に配置し、下記は削除

def get_path(*args, mkdirs=False):
    fps = []
    for fp in args:
        if isinstance(fp, pathlib.Path):
            fps.append(str(fp))
        else:
            fps.append(fp)

    path = os.sep.join(fps)
    path = path.replace('\\', '/')
    path = pathlib.Path(path)

    if mkdirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path

def read(path, **kwargs):
    if not isinstance(path, pathlib.Path):
        path = get_path(path)

    if path.suffix == '.yaml':
        with path.open('r', encoding='utf-8') as r:
            return yaml.safe_load(r, **kwargs)
    elif path.suffix == '.json':
        with path.open('r', encoding='utf-8') as r:
            return json.load(r, **kwargs)
    elif path.suffix == '.csv':
        return pd.read_csv(path, encoding='utf-8', **kwargs)
    elif path.suffix in ('.xls', '.xlsx'):
        return openpyxl.load_workbook(path, **kwargs)
    elif path.suffix in ('.ppt', '.pptx'):
        return Presentation(path, **kwargs)

    raise ValueError(f'Undefined suffix:{path}')

def write(path, data, **kwargs):
    if not isinstance(path, pathlib.Path):
        path = get_path(path, mkdirs=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == '.yaml':
        with path.open(mode='w', encoding='utf-8') as w:
            yaml.dump(data, w, sort_keys=False, allow_unicode=True, **kwargs)
    elif path.suffix == '.json':
        with path.open(mode='w', encoding='utf-8') as w:
            json.dump(data, w, indent=4, sort_keys=False, ensure_ascii=False, **kwargs)
    elif isinstance(data, pd.DataFrame) and path.suffix == '.csv':
        data.to_csv(path, index=False, encoding='utf-8')
    else:
        raise ValueError(f'Undefined suffix:{path}')

class BlobFileIO():
    def __init__(self, blob_container=None, local_dir=None):
        assert (blob_container is not None) != (local_dir is not None), 'Either blob_container or local_dir should be specified.'

        self.blob = Client.get_az_blob_client()
        self.blob_container = blob_container
        self.local_dir = local_dir

    def read(self, rel_path):
        if self.local_dir:
            full_path = get_path(self.local_dir, rel_path)
            return read(full_path)

        cli = self.blob.get_blob_client(container=self.blob_container, blob=rel_path)
        raw = cli.download_blob().readall()

        if rel_path.endswith('.csv'):
            raw = raw.decode('utf-8')
            buf = StringIO(raw)
            return pd.read_csv(buf, encoding='utf-8')

        raise ValueError(f'Undefined suffix:{rel_path}')

    def write(self, rel_path, data):
        if self.local_dir:
            full_path = get_path(self.local_dir, rel_path, mkdirs=True)
            write(full_path, data)
            return

        if isinstance(data, pd.DataFrame) and rel_path.endswith('.csv'):
            data = data.to_csv(index=False, encoding='utf-8')

        cli = self.blob.get_blob_client(container=self.blob_container, blob=rel_path)
        cli.upload_blob(data, overwrite=True)
