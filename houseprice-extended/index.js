const tf = require('@tensorflow/tfjs')
const csvtojson = require('csvtojson')
const path = require('path')

function normalization(value,min,max){
    const norm = (value-min)/(max-min)
    return norm
}

async function loadData(){
    const data = await csvtojson().fromFile(path.resolve(__dirname,'data.csv'))
    const inputs = []
    const labels = []
    const sf = []
    const r = []
    const a = []
    const dic = []

    data.forEach(e=>{
        sf.push(parseFloat(e.SquareFeet))
        r.push(parseFloat(e.Rooms))
        a.push(parseFloat(e.Age))
        dic.push(parseFloat(e.DistanceFromCity))
        labels.push(parseFloat(e.Prices))
    })
    inputs.push(sf,r,a,dic)

    const inputMax = inputs.map(input=>Math.max(...input))
    const inputMin = inputs.map(input=>Math.min(...input))
    const labelMax = Math.max(...labels)
    const labelMin = Math.min(...labels)
    const nsf = sf.map(e=>normalization(e,inputMin[0],inputMax[0]))
    const nr = r.map(e=>normalization(e,inputMin[1],inputMax[1]))
    const na = a.map(e=>normalization(e,inputMin[2],inputMax[2]))
    const ndic = dic.map(e=>normalization(e,inputMin[3],inputMax[3]))
    const ninputs = [nsf,nr,na,ndic]
    const nlabels = labels.map(e=>normalization(e,labelMin,labelMax))

    const xs = tf.tensor2d(ninputs)
    const ys = tf.tensor2d(nlabels,[labels.length,1])
    return {xs,ys,inputMax,inputMin,labelMax,labelMin}
}

// loadData()

async function trainModel(){
    const {xs,ys,inputMax,inputMin,labelMax,labelMin} = await loadData()
    const model = tf.sequential()
    model.add(tf.layers.dense({inputShape: [4],units: 1}))
    model.compile({
        optimizer: tf.train.sgd(0.1),
        loss: 'meanSquaredError'
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

    const inputValue = [2200,4,5,3]
    const ninputValue = inputValue.map((e,i)=>normalization(e,inputMin[i],inputMax[i]))
    const output = model.predict(tf.tensor2d([ninputValue]))
    const answer = await output.data()
    const unNanswer = (answer[0]*(labelMax-labelMin))+labelMin
    console.log(`Answer is ${Math.ceil(unNanswer.toFixed(2))}`)
}

trainModel()