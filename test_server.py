from fastapi import FastAPI
import uvicorn

# Simple test app
app = FastAPI(title="NASA Weather Test API")

@app.get("/")
async def root():
    return {"message": "NASA Weather API is working!", "status": "ok"}

@app.get("/test")
async def test():
    return {"test": "success", "endpoints": ["health", "parameters", "weather/raw", "weather/probability"]}

if __name__ == "__main__":
    print("Starting test server...")
    uvicorn.run(app, host="127.0.0.1", port=8080)