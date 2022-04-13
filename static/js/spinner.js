const btn = document.getElementById("upload");

btn.addEventListener("click", ()=>{

    if(btn.innerText === "YÜKLE"){
        btn.innerText = "Yükleniyor..";
    }else{
        btn.innerText= "GÖNDER";
    }
});
