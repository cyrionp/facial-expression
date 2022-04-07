const btn = document.getElementById("upload");

btn.addEventListener("click", ()=>{

    if(btn.innerText === "GÖNDER"){
        btn.innerText = "Yükleniyor..";
    }else{
        btn.innerText= "GÖNDER";
    }
});
