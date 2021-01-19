const express = require('express');
const app = express();

app.use(express.static('static'));
app.use(express.static('css'));
app.use(express.static('views'));

app.get('/', function(req, res) {
    res.sendFile(__dirname + '/views/index.html');
});

app.get('/monitor', function(req, res) {
    res.sendFile(__dirname + '/views/monitor.html');
});

app.get('/technology', function(req, res) {
    res.sendFile(__dirname + '/views/technology.html');
});

app.get('/future', function(req, res) {
    res.sendFile(__dirname + '/views/future.html');
});

console.log('connected');

app.listen(8080);


