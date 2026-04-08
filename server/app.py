import uvicorn
from api import app

def main():
    # This function is what the autograder is looking for
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()