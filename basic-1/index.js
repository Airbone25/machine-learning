const tf = require('@tensorflow/tfjs');

// DATA
const xs = tf.tensor1d([1,2,3,4])
const ys = tf.tensor1d([3,5,7,9])

// MODEL
const model = tf.sequential()
model.add(tf.layers.dense({inputShape: [1],units: 1}))

model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
})

//TRAIN DATA FUNCTION
async function trainModel(){
    await model.fit(xs,ys,{
        epochs: 1000,
        callbacks:{
            onEpochEnd: (epoch,logs)=>{
                if(epoch%100 == 0){
                    console.log(`Epoch ${epoch}: loss ${logs.loss}`)
                }
            }
        }
    })
    
    // TEST THE MODEL
    const output = model.predict(tf.tensor1d([5]))
    const answer = await output.data()
    console.log(`The answer is ${answer[0].toFixed(2)}`)
}

trainModel()



