const functions = require('firebase-functions');

// // Create and Deploy Your First Cloud Functions
// // https://firebase.google.com/docs/functions/write-firebase-functions
//
// exports.helloWorld = functions.https.onRequest((request, response) => {
//   functions.logger.info("Hello logs!", {structuredData: true});
//   response.send("Hello from Firebase!");
// });
const express = require('express');
const app = express();

app.use(express.static('static'));
app.use(express.static('css'));
app.use(express.static('views'));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname , 'views/index.html'));
});

app.get('/monitor', (req, res) => {
    res.sendFile(_path.join(_dirname,  'views/monitor.html'));
});

app.get('/technology', (req, res) => {
    res.sendFile(path.join(__dirname, 'views/technology.html'));
});

app.get('/future', (req, res) => {
    res.sendFile(path.join(__dirname, '/views/future.html'));
});

console.log('connected');

app.listen(8080);
exports.app = functions.https.onRequest(app)



