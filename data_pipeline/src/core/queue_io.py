import time
from io import StringIO
import pandas as pd
from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
from azure.core.exceptions import ResourceExistsError

from core.util import Conf, dump_as, load_as

class QueueIO():
    def __init__(self, name):
        C = Conf.get()

        self.client = QueueClient.from_connection_string(
            C.storage.connect,
            name,
            message_encode_policy=TextBase64EncodePolicy(),
            message_decode_policy=TextBase64DecodePolicy()
        )

    def read(self, is_dequeue=True):
        queue_msgs = self.client.receive_messages()

        contents = []
        for page in queue_msgs.by_page():
            for msg in page:
                contents.append(msg.content)

                # delete msg
                if is_dequeue:
                    self.client.delete_message(msg)

        return contents

    def write(self, format, msg, size=1):
        assert type(msg) is list, f'invalid type list: {msg}'
        assert size > 0, f'invalid size: {size}'

        try:
            # 非同期処理であるため、queueが作成された場合、sendできるまで待機
            self.client.create_queue()
            time.sleep(5)
        except ResourceExistsError:
            # Queueが存在している
            # 現状queueの存在を確認する方法がないので、exceptで対応
            pass

        if format == 'csv':
            header = msg.pop(0)
            msg_chunks = [msg[i:i + size] for i in range(0, len(msg), size)]
            for rows in msg_chunks:
                rows = '\n'.join([header, *rows])
                self.client.send_message(rows)

        elif format == 'json':
            msg_chunks = [msg[i:i + size] for i in range(0, len(msg), size)]
            for rows in msg_chunks:
                rows = dump_as('json', rows)
                self.client.send_message(rows)

    @staticmethod
    def decode(format, msg):
        m = msg.get_body()
        m = m.decode('utf-8')

        if format == 'csv':
            return pd.read_csv(StringIO(m), encoding='utf-8', dtype=str)
        elif format == 'json':
            return load_as('json', m)
        else:
            return m

    def close(self):
        if hasattr(self.client, 'close'):
            self.client.close()