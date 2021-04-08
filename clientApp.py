from flask import Flask,request,jsonify,render_template
from flask_cors import CORS, cross_origin
from bert import Ner
import preprocess
import re
app = Flask(__name__)
CORS(app)

model = Ner("bert_base_cased_1")
# print(model.state_dict())

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    text = request.json["text"]
    print(text)
    print(type(text))
    try:
        text_cleaned = preprocess.clean_data(text)
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

        words['A1'] = " ".join(a1)#address1
        words['A2'] = " ".join(a2)
        words['A3'] = " ".join(a3)
        words['C'] = " ".join(c)
        words['S'] = " ".join(s)
        words['PS'] = " ".join(ps)
        print(words)
        print(type(words))
        return words
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('127.0.0.1',port=5000)