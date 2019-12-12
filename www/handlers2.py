from models import User
from coroweb import get
import asyncio

@get('/')
async def index(request):
    users = await User.findAll()
    return {
        '__template__': 'test.html',
        'users': users
    }

@get('/api/doclasses')
async def api_do_classes(*,commit):
    the_class = do_classes(commit)
    return dict(classes=the_class)

@get('/api/reply')
async def reply(*,commit):
    commit = prediect_reply(commit)
    return dict(commit=commit)

def prediect_reply(commit):
    return commit


def do_classes(commit):
    return commit