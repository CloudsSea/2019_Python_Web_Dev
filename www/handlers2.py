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
async def api_do_classes(*, comment):
    the_class = do_classes(comment)
    return dict(classes=the_class)

@get('/api/reply')
async def reply(*, comment):
    comment = prediect_reply(comment)
    return dict(comment=comment)

def prediect_reply(comment):
    return comment


def do_classes(comment):
    return comment