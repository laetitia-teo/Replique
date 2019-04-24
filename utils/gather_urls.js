// pull down jquery into the JavaScript console
var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);

// grab urls
var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });
var textToSave = urls.toArray().join('\n');
var a = window.document.createElement('a');
a.href = window.URL.createObjectURL(new Blob([textToSave], {type: "text/plain"}));
a.download = 'urls.txt';

// Append anchor to body.
document.body.appendChild(a);
a.click();

// Remove anchor from body
document.body.removeChild(a);
