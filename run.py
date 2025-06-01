from flaskApp import app # type: ignore
from dotenv import load_dotenv

load_dotenv() 
if __name__ == "__main__":
    app.run(debug=True, port=8080)