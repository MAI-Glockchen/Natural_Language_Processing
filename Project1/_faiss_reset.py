from pathlib import Path
from sqlalchemy import text
from db.session import get_session

# 1) Cleanup DB entries related to FAISS/topic outputs
session = get_session()
try:
    deleted_map = session.execute(text('DELETE FROM faiss_passage_map')).rowcount
    deleted_topic = session.execute(text('DELETE FROM article_topic_outputs')).rowcount
    session.commit()
    print(f'DB cleanup done: faiss_passage_map={deleted_map}, article_topic_outputs={deleted_topic}')
except Exception:
    session.rollback()
    raise
finally:
    session.close()

# 2) Cleanup FAISS-related files
removed = 0
for base in [Path('vector_indices'), Path('vector_indices_db_test')]:
    if not base.exists():
        continue
    for pattern in ('*.faiss', 'summary.json'):
        for p in base.glob(pattern):
            try:
                p.unlink()
                removed += 1
            except Exception as e:
                print(f'WARN could not delete {p}: {e}')

# remove worker subdirs that may contain parallel outputs
for base in [Path('vector_indices'), Path('vector_indices_db_test')]:
    if not base.exists():
        continue
    for wd in base.glob('worker_*'):
        if wd.is_dir():
            for child in wd.rglob('*'):
                if child.is_file():
                    child.unlink(missing_ok=True)
            for child in sorted(wd.rglob('*'), reverse=True):
                if child.is_dir():
                    child.rmdir()
            wd.rmdir()
            removed += 1

print(f'File cleanup done: removed={removed}')
