# funds-search

Search and track job openings at VC funds.

## Running Locally

### Using Python venv

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. Test the health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t funds-search:dev .
   ```

2. Run the container:
   ```bash
   docker run --rm -p 8000:8000 funds-search:dev
   ```

3. Test the health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /search` - Search for job openings at VC funds
