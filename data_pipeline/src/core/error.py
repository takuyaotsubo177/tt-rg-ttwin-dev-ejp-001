class ClientError(Exception):
    pass

class LLMUnexpectedResponse(Exception):
    pass

class SkipRequest(Exception):
    pass
