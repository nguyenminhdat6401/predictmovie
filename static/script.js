fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        title: title,
        year: year,
        runtime: runtime,
        language: language,
        genre: genre
    })
})
.then(res => res.json())
.then(data => {
    console.log(data);
});
