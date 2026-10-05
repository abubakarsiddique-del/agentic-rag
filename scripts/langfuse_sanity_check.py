import os

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=False)

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langchain_groq import ChatGroq

public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
secret_key = os.getenv('LANGFUSE_SECRET_KEY')
host = os.getenv('LANGFUSE_HOST') or os.getenv('LANGFUSE_BASE_URL') or 'https://cloud.langfuse.com'

if not public_key or not secret_key:
    print('FAIL: Langfuse credentials missing')
    raise SystemExit(1)

client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
print('INFO: client initialized', bool(client))

try:
    llm = ChatGroq(
        model='openai/gpt-oss-120b',
        temperature=0,
        max_retries=1,
        callbacks=[CallbackHandler()],
    )
    with client.start_as_current_observation(name='sanity-check', as_type='chain'):
        with propagate_attributes(user_id='sanity-user', session_id='sanity-session', tags=['langfuse-sanity']):
            response = llm.invoke('Reply with exactly: OK')
            print('RESPONSE:', getattr(response, 'content', response))
            trace_id = client.get_current_trace_id()
            print('TRACE_ID:', trace_id)
            if trace_id:
                print('TRACE_URL:', client.get_trace_url(trace_id=trace_id))

    client.flush()
    if not trace_id:
        raise RuntimeError('No Langfuse trace ID was created')
    print('SUCCESS: Langfuse reached and flushed a trace')
except Exception as exc:
    status_code = getattr(exc, 'status_code', None)
    print('FAIL:', type(exc).__name__, f'(status={status_code})' if status_code else '')
    raise
