# indexer.py

from azure.search.documents.indexes.models import (
    HnswParameters,
    HnswVectorSearchAlgorithmConfiguration,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
)

from core.util import Client

ANALYZER = {
    'en': 'en.microsoft',
    'ja': 'ja.microsoft',
    '_default': 'en.microsoft',
}
DIMENSION = {
    'text-embedding-3-large': 3072,
}

class ExpertTwinIndexer():
    def __init__(self, index_name, lang, semantic_name, embed_model):
        self.index_name = index_name
        self.lang = lang
        self.semantic_name = semantic_name
        self.embed_model = embed_model

    def __call__(self):
        client = Client.get_index_client()
        if self.index_name in [n for n in client.list_index_names()]:
            return

        analyzer = ANALYZER.get(self.lang, ANALYZER['_default'])
        embed_dim = DIMENSION[self.embed_model]

        fields = [
            SimpleField(name='id', type='Edm.String', key=True),
            SimpleField(name='category', type='Edm.String', filterable=True),       # ユーザーのロールによってアクセスできるcategoryを設定しておき、アクセス制限する
            SearchableField(name='title', type='Edm.String', analyzer_name=analyzer, searchable=True),
            SearchableField(name='content', type='Edm.String', analyzer_name=analyzer, searchable=True),
            SearchableField(name='keywords', type='Edm.String', analyzer_name=analyzer, searchable=True),
            SimpleField(name='source_fn', type='Edm.String', filterable=True),      # ファイル名
            SimpleField(name='source', type='Edm.String', filterable=True),         # ファイルのパス（ファイル内の識別番号を含む）
            SimpleField(name='date', type='Edm.DateTimeOffset', filterable=True, sortable=True),
            SimpleField(name='importance', type='Edm.Double', filterable=True, sortable=True, facetable=True),
            SimpleField(name='importance_reason', type='Edm.String'),
            SearchField(
                name='embedding',
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                hidden=False,
                searchable=True,
                filterable=False,
                sortable=False,
                facetable=False,
                vector_search_dimensions=embed_dim,
                vector_search_profile='embedding_config',
            ),
        ]

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            semantic_settings=SemanticSettings(
                configurations=[
                    SemanticConfiguration(
                        name=self.semantic_name,
                        prioritized_fields=PrioritizedFields(
                            title_field=SemanticField(field_name='title'),
                            prioritized_content_fields=[
                                # body, summary, introduction
                                SemanticField(field_name='content'),
                            ],
                            prioritized_keywords_fields=[
                                # tags,  keywords, meta_description
                                SemanticField(field_name='keywords'),
                            ]
                        ),
                    )
                ]
            ),
            vector_search=VectorSearch(
                algorithms=[
                    HnswVectorSearchAlgorithmConfiguration(
                        name='hnsw_config',
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters=HnswParameters(metric='cosine'),
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name='embedding_config',
                        algorithm='hnsw_config',
                    ),
                ],
            ),
        )
        client.create_index(index)
