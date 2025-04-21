const tf = require('@tensorflow/tfjs');
const csvtojson = require('csvtojson')
const path = require('path')

function buildVocabulary(sentences){
    const vocab = new Set()
    sentences.forEach(s=>{
        s.split(" ").forEach(w=>vocab.add(w))
    })
    const wordtoindex = {}
    Array.from(vocab).forEach((w,i)=>{
        wordtoindex[w] = i+1
    })
    return wordtoindex
}

function tokenize(sentence,wordtoindex,maxLen){
    const tokens = sentence.split(" ").map(w=>wordtoindex[w] || 0)
    while(tokens.length<maxLen){
        tokens.push(0)
    }
    return tokens.slice(0,maxLen)
}

function buildIndexToWord(word2index) {
    const index2word = {}
    Object.entries(word2index).forEach(([word, idx]) => {
      index2word[idx] = word
    })
    return index2word
}

function detokenize(tokens, index2word) {
    return tokens
      .map(token => index2word[token] || '') 
      .filter(word => word !== '')             
      .join(' ')
}

async function loadData(){
    const data = await csvtojson().fromFile(path.resolve(__dirname,'data.csv'))
    const inputs = data.map(e=>e.Input)
    const outputs = data.map(e=>e.Output)
    const inputsVocab = buildVocabulary(inputs)
    const outputsVocab = buildVocabulary(outputs)
    const tokenizedInputs = inputs.map(e=>tokenize(e,inputsVocab,5))
    const tokenizedOutputs = outputs.map(e=>tokenize(e,outputsVocab,5))

    const xs = tf.tensor2d([tokenizedInputs])
    const ys = tf.tensor2d([tokenizedOutputs])
    return { xs,ys,inputsVocab }
}

async function trainModel(){
    const { xs,ys } = await loadData()

    const model = tf.sequential()
    model.add(tf.layers.dense({inputShape: [5],units: 7,activation: 'relu'}))
    model.add(tf.layers.dense({units: 7,activation: 'softmax'}))
    model.compile({
        optimizer: tf.train.adam,
        loss: 'sparseCategoricalCrossentropy',
    })
    model.fit(xs,ys,{
        epochs: 1000,
        callbacks: {
            onEpochEnd: (epoch,logs)=>{
                if(epoch%100==0){
                    console.log(`Epoch ${epoch}: Loss ${logs.loss}`)
                }
            }
        }
    })

    const string = "Hi"
    const resVocab = buildVocabulary()
    const inputToken = resVocab.map(e=>tokenize(string,resVocab,5))
    const inputVector = tf.tensor2d([inputToken])
    const output = model.predict(inputVector)
    const answer = await output.data()
    console.log(answer)
}

trainModel()
