import re
import time
import random
import uuid
import fitz
import shutil
import base64
from abc import ABC, abstractmethod
from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from openai import RateLimitError

from core.util import Conf, Client, norm, get_path, load_as, dump_as, get_logger
from core.error import LLMUnexpectedResponse, SkipRequest

LLM_JSON_INVALID_PTNS = [
    re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
]

logger = get_logger(__name__)

# -------------------------------------------------- LLM
class SingleAgent(ABC):
    def __init__(self, template, gpt='gpt41', output_parser='str', retry=2, wait_sec_max=3):
        C = Conf.get()
        self.template = template
        self.output_parser = output_parser
        self.retry = retry
        self.wait_sec_max = wait_sec_max

        if gpt == 'gpt41':
            self.model_name = C.openai.gpt41.name
        else:
            self.model_name = C.openai.gpt41_mini.name

        self.prompt = None
        self.llm = None
        self.parser = None

    def __call__(self, doc):
        kwargs = self._get_kwargs(doc)
        if not kwargs:
            return
        elif type(kwargs) is not list:
            kwargs = [kwargs]

        self._activate_chain_module()
        retry = self.retry
        chain = self.prompt | self.llm | self.parser

        for kw in kwargs:
            while True:
                output = chain.invoke(kw)
                try:
                    self._set_output(doc, output)
                    return
                except LLMUnexpectedResponse as e:
                    retry -= 1
                    if retry > 0:
                        time.sleep(random.randint(1, self.wait_sec_max))
                        continue
                    raise e

    def _activate_chain_module(self):
        if self.llm:
            return

        C = Conf.get()
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = AzureChatOpenAI(
            openai_api_version=C.openai.version,
            azure_deployment=self.model_name,
            model_name=self.model_name,
            azure_endpoint=C.openai.endpoint,
            api_key=C.openai.key
        )
        self.parser = JsonOutputParser() if self.output_parser == 'json' else StrOutputParser()

    @abstractmethod
    def _get_kwargs(self, doc):
        pass

    @abstractmethod
    def _set_output(self, doc, output):
        pass


class CreateProjectSummaryAgent(SingleAgent):
    def __init__(self, input_keys=['_category', '_theme', '_industry', '_overview'], output_by='page_content'):
        self.input_keys = input_keys
        self.output_by = output_by
        super().__init__(template="""
###Instruction###
入力されたデータは、コンサルティング会社における生成AIに関するプロジェクトの実績データです。
このデータを以下の形式で文章化してください：
生成AIに関する<カテゴリ>のプロジェクト実績。<業界>業界において、<テーマ>の支援を実施。<概要/支援内容>を行った。

###User###
カテゴリ: {category}
テーマ: {theme}
業界: {industry}
概要/支援内容: {overview}

###Assistant###
""")

    def _get_kwargs(self, doc):
        m = doc.__dict__
        kwargs = {}
        for k in self.input_keys:
            v = m.get(k, m['metadata'].get(k))
            k_without_underscore = k.lstrip('_')
            if not v:
                kwargs[k_without_underscore] = ""
            else:
                kwargs[k_without_underscore] = norm(v, 'neologdn')

        # 有効なコンテンツがあるか確認
        if not any(kwargs.values()):
            return None

        return kwargs

    def _set_output(self, doc, output):
        if not output:
            raise LLMUnexpectedResponse('Empty')

        if len(output) == len(output.encode('utf-8')):
            raise LLMUnexpectedResponse(f'Not japanese: {output}')

        if self.output_by == 'page_content':
            doc.page_content = output
        else:
            doc.metadata[self.output_by] = output


class CreateSemanticKeyAgent(SingleAgent):
    def __init__(self, input_keys=['page_content'], output_by={'title': 'title', 'keywords': 'keywords'}):
        self.input_keys = input_keys
        self.output_by = output_by
        super().__init__(template="""
###Instruction###
入力された文章から、下記に示す項目のmeta情報を生成し、JSON形式で出力します。
- title: 文章全体の意味を考慮し、タイトルを生成する
- keywords: 文章全体から、検索時に使用されそうなキーワードを抽出する

###User###
AI の研究開発や利活用は、今後急速に進展することが期待されているところであり、AI ネットワーク化が進展していく過程で、個人、地域社会、各国、
国際社会の抱える様々な課題の解決に大きく貢献するなど、人間及びその社会や経済に多大な便益を広範にもたらすことが期待される

###Assistant###
{{
  "title": "AI研究開発による個人・社会貢献の期待",
  "keywords": [
    "AI",
    "研究開発",
    "利活用",
    "ネットワーク化",
    "社会貢献",
    "経済効果",
    "課題解決",
    "地域社会",
    "国際社会"
  ]
}}

###User###
{input}

###Assistant###
""")

    def _get_kwargs(self, doc):
        m = doc.__dict__
        vs = []
        for k in self.input_keys:
            v = m.get(k, m['metadata'].get(k))
            if not v:
                continue

            v = norm(v, 'neologdn')
            vs.append(v)

        if not vs:
            return None

        return { 'input': '\n'.join(vs) }

    def _set_output(self, doc, output):
        # valid 1/1: empty
        if not output:
            raise LLMUnexpectedResponse('Empty')

        # valid 2/3: japanese
        if len(output) == len(output.encode('utf-8')):
            raise LLMUnexpectedResponse(f'Not japanese: {output}')

        # valid 3/3: json format
        try:
            for p in LLM_JSON_INVALID_PTNS:
                m = p.search(output)
                if m:
                    output = m.group(1)
                    break

            meta = load_as('json', output)

            # title
            t = meta.get('title', None)
            meta['title'] = t
            assert t, f'title must be required key: {output}'

            # keywords
            kws = meta.get('keywords', None)
            if type(kws) is not list:
                kws = [kws]

            meta['keywords'] = ' '.join(kws)
            assert meta['keywords'], f'keywords[N] must be required key: {output}'

        except Exception as e:
            raise LLMUnexpectedResponse(f'invalid LLM output json format {e}, {output}')

        # success
        for frm,to in self.output_by.items():
            doc.metadata[to] = meta[frm]

        doc.metadata['keywords'] = ' '.join(kws)


class ScoreImportanceAgent(SingleAgent):
    def __init__(self, input_keys=['page_content'], output_by={'importance': 'importance'}):
        self.input_keys = input_keys
        self.output_by = output_by
        super().__init__(
            template="""ユーザー入力は、匠（経験豊富なエキスパート）の経験や考察を記した文章です。  
この文章が**当人の専門知識・実践経験・独自視点や思考パターン**を理解するうえでどれほど重要か、0～100の整数で評価してください。

## 匠のペルソナ
{persona}

## 評価観点
次の3点を総合的に判断してください。
1. **専門知識の深さ**：高度で一般的でない知識や技術が示されているか
2. **実践経験の具体性**：実際の業務や事例に基づく具体的な記述があるか
3. **独自の視点・思考**：他の専門家と異なる独自の切り口やアプローチがあるか

## 評価基準（0-100点）
- **0-20点（一般論）**  
  - 基本的な情報のみ／個人の経験や専門性が感じられない／誰でも知っている内容
- **21-40点（初級専門）**  
  - 基本的な専門用語や概念はあるが、深い洞察や独自性に乏しい
- **41-60点（中級専門）**  
  - 具体的な事例や経験があり、専門知識も含まれるが業界では一般的
- **61-80点（上級専門）**  
  - 豊富な実務経験や独自の手法・視点が明確に示されている
- **81-100点（匠レベル）**  
  - 独自のフレームワークや理論、深い洞察、革新的な視点、その人にしか語れない知見が含まれている

## 出力形式
以下のJSON形式で回答してください:
{{
  "reason": "[評価理由を簡潔に説明]",
  "score": [重要度スコア(0-100の整数)]
}}

## 評価対象の文章
{input}
""",
            output_parser='json'
        )

    def _get_kwargs(self, doc):
        C = Conf.get()
        persona = C.persona
        persona_parts = []
        if persona.get('name'):
            persona_parts.append(f"- 名前: {persona['name']}")
        # 会社名は必須
        persona_parts.append(f"- 会社: {persona['company']}")
        if persona.get('department'):
            persona_parts.append(f"- 部署: {persona['department']}")
        if persona.get('position'):
            persona_parts.append(f"- 役職: {persona['position']}")
        if persona.get('responsibility'):
            persona_parts.append(f"- 役割: {persona['responsibility']}")
        if persona.get('specialty'):
            persona_parts.append(f"- 専門分野: {persona['specialty']}")

        m = doc.__dict__
        vs = []
        for k in self.input_keys:
            v = m.get(k, m['metadata'].get(k))
            if not v:
                continue
            v = norm(v, 'neologdn')
            vs.append(v)
        if not vs:
            return None
        return {
            'input': '\n'.join(vs),
            'persona': '\n'.join(persona_parts)
        }

    def _set_output(self, doc, output):
        if not output:
            raise LLMUnexpectedResponse('Empty')
            
        # JSONレスポンスから重要度スコアを取得
        if 'score' in output:
            # 呼び出し元の期待する形式に合わせてメタデータに設定
            for key, metadata_key in self.output_by.items():
                doc.metadata[metadata_key] = output['score']
            
            if 'reason' in output:
                doc.metadata['importance_reason'] = output['reason']
        else:
            raise LLMUnexpectedResponse(f'Missing score in JSON response: {output}')


# -------------------------------------------------- VLM
class SingleGptImgAgent(ABC):
    def __init__(self, template, retry=2, wait_sec_max=3):
        self.template = template
        self.retry = retry
        self.wait_sec_max = wait_sec_max

    def __call__(self, doc):
        kwargs = self._get_kwargs(doc)
        if not kwargs:
            return

        C = Conf.get()
        client = Client.get_az_openai_client()
        retry = self.retry

        while True:
            output = client.chat.completions.create(
                model=C.openai.gpt41.name,
                temperature=0.,
                **kwargs
            )

            try:
                self._set_output(doc, output.choices[0].message.content)
                return
            except LLMUnexpectedResponse as e:
                retry -= 1
                if retry > 0:
                    time.sleep(random.randint(1, self.wait_sec_max))
                    continue
                raise e

    @abstractmethod
    def _get_kwargs(self, doc):
        pass

    @abstractmethod
    def _set_output(self, doc, output):
        pass


class OCRWholeSlideByGptAgent(SingleGptImgAgent):
    def __init__(self):
        super().__init__(template="""
# 役割
あなたは、ロジカルシンキングとプレゼンテーション設計に優れたコンサルタントです。
提供されるスライド（ページごとの画像）について、OPQフレームワークとピラミッド構造の観点から構成を分析し、その論理展開を評価・要約するレポートを作成します。
このレポートは、資料レビューや改善、今後の資料作成の参考に資することを目的とします。

# 出力形式

## 1. 概要
- **主題**: 何をテーマとしたスライドか。
- **OPQ分析**:
    - **Objective** (読み手の目標): 読み手が目指している状況や理想の状態。
    - **Problem** (現状とのギャップ): Objectiveと現状との間にある問題・課題。
    - **Question** (読み手の疑問): Problemを解決するために読み手が抱くであろう中心的な問い。
- **Answer** (主メッセージ): 上記のQuestionに対する、このスライドが提示する明確な答え・結論。
- **ストーリーライン**: 
    OPQの提示から主メッセージ（Answer）へ、そしてそれを支える根拠や具体的なアプローチへと、読み手の疑問を解消し、納得感を醸成する流れがどのように構築されているかを簡潔に評価する。

## 2. ページ別要約
各ページについて：
- **ページ番号 - タイトル**
    - **役割**: このページが全体構成（OPQ提示、主メッセージ、根拠、具体策など）の中で果たしている役割を一言で。
    - **要点**（主張と根拠）: ページ内の主張と、それを支える根拠・具体例の関係性が分かるように記述する。

# ルール
- マークダウン形式で出力
- 2000文字程度
- 全体構成と論理展開に焦点を当てる
- 詳細なテキストではなく要点を記述
- 前置きや締めの文言は不要
""")

    def _get_kwargs(self, doc):
        img_b64_list = doc.metadata.pop('_source_as_img_b64_list', None)
        if not img_b64_list or len(img_b64_list) == 0:
            raise SkipRequest('Not found _source_as_img_b64_list or empty list')

        content = []
        for i, img_b64 in enumerate(img_b64_list):
            content.append(
                {
                    'type': 'text',
                    'text': f"\n--- p{i+1} ---"
                }
            )
            content.append(
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{img_b64}'
                    }
                }
            )

        return {
            'messages': [
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'text',
                            'text': self.template
                        },
                    ]
                },
                {
                    'role': 'user',
                    'content': content
                }
            ]
        }

    def _set_output(self, doc, output):
        # valid 1/3: empty
        if not output:
            raise LLMUnexpectedResponse('Empty')

        # valid 2/3: status
        if 'SKIP:' in output:
            raise SkipRequest('Not found content')

        # valid 3/3: japanese
        if len(output) == len(output.encode('utf-8')):
            raise LLMUnexpectedResponse(f'Not japanese: {output}')

        # ok
        doc.page_content = output


class OCRSinglePageByGptAgent(SingleGptImgAgent):
    def __init__(self):
        super().__init__(template="""
# 役割
プレゼンテーションスライド1ページの詳細構成を分析し、スライドレビューや資料作成時の参考となるお手本分析を作成する。
提供された1ページの画像を確認し、レイアウト・内容構成・視覚的要素を中心に詳細に整理すること。
分析時は下記の体裁・デザインルールを参考にすること。

## スライド体裁・デザインルール
- 段落前は0pt、段落後は6pt、行間は1行とする
- フォントは游ゴシック、サイズは12pt以上（エグゼクティブ向けはさらに大きく）
- ブレットは使用し、文字で「・」を打たない。ブレットと枠線/文頭の幅も調整（ルーラー/Tabを使用）
- 改行位置は自ら調整し、単語や節の切れ目で行う。自動折り返しは使用しない
- テキストボックスの「テキストに合わせて図形のサイズを調整する」は絶対に使わない
- 色は意味合いを持たせて使い、各色の特徴を考慮する。構造上の分類・粒度の差・強調など用途を明確に
- 彩度は各オブジェクトの意味合いに応じて調整し、ビビッドや原色は避ける
- コントラストは暗×明の組み合わせを推奨し、暗×暗・明×明は避ける
- 図形の形状・サイズ・位置は意味合いごとに統一し、小数点以下も調整する
- 四角形はMECE、矢印は論理や検討の遷移、矢羽はプロセスや前提条件・インプット、円はラベリングや性質の違いを示す際に使用
- 丸はイコールや詳細説明、三角は方向性や論理展開など意味合いで使い分ける
- スライド作成前に必ずロジックツリーを考え、伝えたいメッセージの根拠を2〜5個に分解（初級者〜中級者は帰納法推奨）、分解したものをボディに記載
- スライドは原則、縦か横に分解して記載

# 出力形式
## 1. 基本情報
- **タイトル**:
- **スライドの目的**:
- **対象セクション**:

## 2. 構成要素分析
- **レイアウト構造**
    - 全体レイアウトの特徴
    - 要素配置の工夫
- **コンテンツ構成**
    - 主要メッセージ
    - 情報の階層化
    - 論点整理方法
- **視覚的要素**
    - 図表・画像の活用
    - 色彩・フォントの使い方
    - 視認性への配慮

## 3. お手本ポイント
- **効果的な点**
- **参考になる技法**
- **応用可能な要素**

# ルール
- マークダウン形式で出力
- 1000文字以内
- 1ページの詳細分析に焦点を当てる
- スライド制作の参考となる具体的な要素を記述
- 前置きや締めの文言は不要
- 他のスライド作成時の手本となる内容にする
""")

    def _get_kwargs(self, doc):
        img_b64 = doc.metadata.pop('_source_as_img_b64', None)
        page_num = doc.metadata.get('_page_num', None)
        whole_content = doc.metadata.get('_whole_content', None)
        if not img_b64:
            raise SkipRequest('Not found _source_as_img_b64')

        content = [
            {
                'type': 'text',
                'text': f"このスライドはページ{page_num + 1 if page_num is not None else 'X'}です。詳細に解析してください。"
            }
        ]
        content.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/png;base64,{img_b64}'
            }
        })
        if whole_content:
            content.append({
                'type': 'text',
                'text': f"スライド全体の構成情報：\n\n{whole_content}"
            })
        return {
            'messages': [
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'text',
                            'text': self.template
                        },
                    ]
                },
                {
                    'role': 'user',
                    'content': content
                }
            ]
        }

    def _set_output(self, doc, output):
        # valid 1/3: empty
        if not output:
            raise LLMUnexpectedResponse('Empty')

        # valid 2/3: status
        if 'SKIP:' in output:
            raise SkipRequest('Not found content')

        # valid 3/3: japanese
        if len(output) == len(output.encode('utf-8')):
            raise LLMUnexpectedResponse(f'Not japanese: {output}')

        # ok
        doc.page_content = output


# -------------------------------------------------- API
class TextEmbeddingAgent():
    def __init__(self, input_keys=['page_content']):
        self.input_keys = input_keys
        self.retry = 2
        self.wait_sec_max = 3

    def __call__(self, doc):
        kwargs = self._get_kwargs(doc)
        if not kwargs:
            return

        client = Client.get_az_openai_client()
        retry = self.retry

        while True:
            try:
                output = client.embeddings.create(**kwargs)
                self._set_output(doc, output)
                return

            except RateLimitError as e:
                retry -= 1
                if retry > 0:
                    time.sleep(random.randint(1, self.wait_sec_max))
                raise e

    def _get_kwargs(self, doc):
        C = Conf.get()

        if 'embedding' not in doc.metadata:
            return None

        m = doc.__dict__
        vs = []
        for k in self.input_keys:
            v = m.get(k, m['metadata'].get(k))
            if not v:
                continue

            v = norm(v, 'neologdn')
            vs.append(v)

        if not vs:
            return None

        embed_input = '\n'.join(vs)
        return {
            'input': [embed_input],
            'model': C.openai.embed.name
        }

    def _set_output(self, doc, output):
        doc.metadata['embedding'] = output.data[0].embedding

class CreateIndexAgent():
    def __init__(self, index_name, insert_meta_keys):
        self.index_name = index_name
        self.insert_meta_keys = insert_meta_keys

    def __call__(self, doc):
        insert_kwargs = self._get_insert_kwargs(doc)
        if not insert_kwargs:
            return

        client = Client.get_az_search_client(self.index_name)
        client.upload_documents(**insert_kwargs)

    def _get_insert_kwargs(self, doc):
        m = doc.metadata
        return {
            'documents': [
                {
                    'id': m.get('id', str(uuid.uuid4())),
                    'content': doc.page_content,
                    **{ k:m[k] for k in self.insert_meta_keys }
                }
            ]
        }


# -------------------------------------------------- Other
class DocToImgAgent():
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    def __call__(self, doc):
        fp = doc.metadata.pop('_local_path')
        dst_dir = self.tmp_dir / fp.stem
        dst_dir.mkdir(parents=True, exist_ok=True)

        # TODO: pdf以外の拡張子に対応
        if fp.suffix == '.pdf':
            img_paths = self._pdf_to_img(fp, dst_dir)
        else:
            raise ValueError(f'Unexpected extension: {fp}')

        doc.metadata['_img_paths'] = img_paths

    def _pdf_to_img(self, file_path, dst_dir):
        pdf = fitz.open(file_path)
        img_paths = []

        try:
            for i in range(pdf.page_count):
                page = pdf.load_page(i)
                pix = page.get_pixmap()

                fp = dst_dir / f'p{i}.png'
                pix.save(str(fp))
                img_paths.append(fp)

        except Exception as e:
            # 想定外のエラーが発生したため、途中状態のoutputは全て削除する
            shutil.rmtree(dst_dir)
            raise e

        finally:
            pdf.close()

        return img_paths


class DocImgListToTextAgent():
    def __init__(self, ocr_agent, max_images=50):
        self.ocr_agent = ocr_agent
        self.max_images = max_images
        
    def __call__(self, doc_base):
        img_paths = doc_base.metadata.pop('_img_paths')
        img_dir = img_paths[0].parent
        
        if len(img_paths) > self.max_images:
            logger.warning(f"Too many images ({len(img_paths)}), limiting to {self.max_images}")
            img_paths = img_paths[:self.max_images]
        
        img_b64_list = []
        
        try:
            for fp in img_paths:
                with open(fp, 'rb') as img:
                    img_b64 = base64.b64encode(img.read()).decode('utf-8')
                    img_b64_list.append(img_b64)
            
            doc = Document(
                page_content='',
                metadata={
                    '_source_as_img_b64_list': img_b64_list,
                },
            )
            
            self.ocr_agent(doc)
            doc_base.page_content = doc.page_content
            
        except SkipRequest:
            raise SkipRequest('Content processing was skipped')
        
        except Exception as e:
            if img_dir.exists():
                shutil.rmtree(img_dir)
            raise e


class DocImgToTextAgent():
    def __init__(self, ocr_agent, cleanup_dir=True):
        self.ocr_agent = ocr_agent
        self.cleanup_dir = cleanup_dir

    def __call__(self, doc_base):
        img_paths = doc_base.metadata.pop('_img_paths')
        img_dir = img_paths[0].parent
        
        # ページ番号を取得して対応する画像を選択
        page_num = doc_base.metadata.get('_page_num', 0)
        img_path = img_paths[page_num] if page_num < len(img_paths) else img_paths[0]
        
        try:
            with open(img_path, 'rb') as img:
                img_b64 = base64.b64encode(img.read()).decode('utf-8')
            
            doc = Document(
                page_content='',
                metadata={
                    '_source_as_img_b64': img_b64,
                },
            )
            self.ocr_agent(doc)
            doc_base.page_content = doc.page_content
            
        except SkipRequest:
            pass
            
        except Exception as e:
            if img_dir.exists():
                shutil.rmtree(img_dir)
            raise e
            
        finally:
            if self.cleanup_dir and img_dir.exists():
                shutil.rmtree(img_dir)