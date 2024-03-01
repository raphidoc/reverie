from reverie.image.revecube import ReveCube
import logging

# From ACOLITE
# # update version info
# if 'version' in config:
#     version = 'Generic Version {}'.format(config['version'])
# else:
#     version = 'Generic GitHub Clone'
#
#     gitdir = '{}/.git'.format(path)
#     gd = {}
#     if os.path.isdir(gitdir):
#         gitfiles = os.listdir(gitdir)
#
#         for f in ['ORIG_HEAD', 'FETCH_HEAD', 'HEAD']:
#             gfile = '{}/{}'.format(gitdir, f)
#             if not os.path.exists(gfile): continue
#             st = os.stat(gfile)
#             dt = datetime.datetime.fromtimestamp(st.st_mtime)
#             gd[f] = dt.isoformat()[0:19]
#
#         version_long = ''
#         if 'HEAD' in gd:
#             version_long+='clone {}'.format(gd['HEAD'])
#             version = 'Generic GitHub Clone c{}'.format(gd['HEAD'])
#         if 'FETCH_HEAD' in gd:
#             version_long+=' pull {}'.format(gd['FETCH_HEAD'])
#             version = 'Generic GitHub Clone p{}'.format(gd['FETCH_HEAD'])

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
