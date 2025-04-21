const tf = require('@tensorflow/tfjs');
const csvtojson = require('csvtojson');
const fs = require('fs');
const path = require('path');

function normalization(value,max,min){
    const norm = (value-min)/(max-min)
    return norm
}

async function loadData(){
    const data = await csvtojson().fromFile(path.resolve(__dirname,'data.csv'))
    const inputs = []
    const labels = []
    data.forEach(item=>{
        inputs.push(parseFloat(item.SquareFeet))
        labels.push(parseFloat(item.Prices))
    })

    const inputMax = Math.max(...inputs)
    const inputMin = Math.min(...inputs)
    const labelMax = Math.max(...labels)
    const labelMin = Math.min(...labels)

    const normInput = inputs.map(x=>normalization(x,inputMax,inputMin))
    const normLabel = labels.map(y=>normalization(y,labelMax,labelMin))

    const xs = tf.tensor2d(normInput,[inputs.length,1])
    const ys = tf.tensor2d(normLabel,[labels.length,1])
    return { xs,ys,inputMax,inputMin,labelMax,labelMin }
}

async function trainModel(){
    const {xs,ys,inputMax,inputMin,labelMax,labelMin} = await loadData()

    const model = tf.sequential()
    model.add(tf.layers.dense({inputShape: [1],units: 1}))
    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.sgd(0.1)
    })

    await model.fit(xs,ys,{
        epochs: 2000,
        callbacks: {
            onEpochEnd: (epoch,logs)=>{
                if(epoch%100==0){
                    console.log(`Epoch ${epoch}: Loss ${logs.loss}`)
                }
            }
        }
    })

    const inputValue = normalization(3000,inputMax,inputMin)
    const output = model.predict(tf.tensor2d([[inputValue]]))
    const answer = await output.data()
    const unNormAnswer = (answer[0]*(labelMax-labelMin))+labelMin
    console.log(Math.ceil(unNormAnswer.toFixed(2)))
}

trainModel()

