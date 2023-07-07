

import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

import { ControllerDataset } from './controller_dataset';
import * as ui from './ui';


const NUM_CLASSES = 4;

let webcam;

const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet;
let model;


async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}


ui.setExampleHandler(async label => {
  let img = await getImage();

  controllerDataset.addExample(truncatedMobileNet.predict(img), label);

  // Draw the preview thumbnail.
  ui.drawThumb(img, label);
  img.dispose();
})


async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }


  model = tf.sequential({
    layers: [

      tf.layers.flatten(
        { inputShape: truncatedMobileNet.outputs[0].shape.slice(1) }),

      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),

      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });


  const optimizer = tf.train.adam(ui.getLearningRate());

  model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });


  const batchSize =
    Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
      `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }


  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {

    const img = await getImage();


    const embeddings = truncatedMobileNet.predict(img);


    const predictions = model.predict(embeddings);


    const predictedClass = predictions.as1D().argMax();
    const classId = (await predictedClass.data())[0];
    img.dispose();

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}


async function getImage() {
  const img = await webcam.capture();
  const processedImg =
    tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  img.dispose();
  return processedImg;
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  ui.startPacman();
  isPredicting = true;
  predict();
});

async function init() {
  try {
    webcam = await tfd.webcam(document.getElementById('webcam'));
  } catch (e) {
    console.log(e);
    document.getElementById('no-webcam').style.display = 'block';
  }
  truncatedMobileNet = await loadTruncatedMobileNet();

  ui.init();


  const screenShot = await webcam.capture();
  truncatedMobileNet.predict(screenShot.expandDims(0));
  screenShot.dispose();
}

init();
