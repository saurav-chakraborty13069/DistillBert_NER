from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


from pydantic import BaseModel
from typing import List

from bert import Ner
import preprocess
import uvicorn



app = FastAPI()


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

model = Ner("bert_base_cased_1")

class TextSample(BaseModel):
    text: str

# class RequestBody(BaseModel):
#     samples: List[text]

@app.get("/", response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/test")
async def test():
    return {"message":"test1","message2":"test2"}

@app.post('/predict', response_class=HTMLResponse)
async def predict(text: TextSample, request:Request):
    # text = request.json["text"]
    text = str(text)
    text_str = text.split('=')[1].replace("'","")
    # print('text is  :',text_str)
    # print(type(text_str))
    try:
        text_cleaned = preprocess.clean_data(text_str)
        # print(text_cleaned)
        out = model.predict(text_cleaned)
        # print(type(out))
        # return jsonify({"result":out})
        words = {}
        a1 = []
        a2 = []
        a3 = []
        s = []
        c = []
        ps = []
        # print(out)
        for item in out:

            tag = item['tag'].split('-')
            word = item['word']

            if len(tag) == 2:
                if tag[1] == 'A1':
                    # print(word)
                    a1.append(word)
                    # words['A1'] = a1.append(word)
                elif tag[1] == 'A2':
                    a2.append(word)
                    # words['A2'] = a2.append(word)
                elif tag[1] == 'A3':
                    a3.append(word)
                    # words['A3'] = a3.append(word)
                elif tag[1] == 'C':
                    c.append(word)
                    # words['C'] = c.append(word)
                elif tag[1] == 'S':
                    s.append(word)
                    # words['S'] = s.append(word)
                elif tag[1] == 'PC':
                    ps.append(word)
                # words['PS'] = ps.append(word)

        words['A1'] = " ".join(a1)  # address1
        words['A2'] = " ".join(a2)
        words['A3'] = " ".join(a3)
        words['C'] = " ".join(c)
        words['S'] = " ".join(s)
        words['PS'] = " ".join(ps)
        # print(words)
        json_compatible_item_data = jsonable_encoder(words)
        return JSONResponse(content=json_compatible_item_data)

    except Exception as e:
        print(e)
        return {"result": "Model Failed"}



if __name__=='__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)