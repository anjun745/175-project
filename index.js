const api = require('realtime-newsapi')();

api.on('articles', (articles) => {
  console.log(articles);
});