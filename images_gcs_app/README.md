Service to interact with GCS and the database stored mapping image_id to GCS's blob name.

To run this service expects 2 env vars:
1. `GCS_BUCKET` - name of bucket on GCS
2. `IMAGES_GCS_API_PG_DNS` - full url to database, see example in `.env_example`


Example of run locally:
`uvicorn app:app --reload --port 8002 --host 0.0.0.0`
