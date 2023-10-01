document.addEventListener("DOMContentLoaded", function() {
    // Get the video element
    const video = document.getElementById("video");
    const direction = document.getElementById("direction");
    const name = document.getElementById("name");
    let approved = [0,0,0,0] 

    // Check if the browser supports getUserMedia
    var socket = io.connect("https://" + document.domain + ":" + location.port);
    socket.connect();
    const imgElement = document.getElementById("image-element");
    socket.on("data", function(data) {
        if ('img' in data){
        imgElement.src = 'data:image/png;base64, ' + data.img
        direction.innerText = data.direction
        }
        // console.log(Object.keys(data));

    });
    w8 = true
    socket.on("approved", function(data) {
        if (w8){
            video.pause()
            name.innerHTML = "processing"
        }
        else{
            approved[data] = 1
            video.play()

        }
        // console.log(Object.keys(data));

    });
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Access the camera
        navigator.mediaDevices
            .getUserMedia({
                video: true,
            })
            .then((stream) => {
                video.srcObject = stream;
            })
            .catch((error) => {
                console.error("Error accessing the camera:", error);
            });

        var canvas = document.createElement("canvas")
        canvas.width = 400; // Set the desired width
        canvas.height = 300; // Set the desired height
        const context = canvas.getContext("2d");

        function captureAndSend() {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const frameDataUrl = canvas.toDataURL("image/jpeg");
            console.log(name.value);
            socket.emit("get_image", {"img":frameDataUrl,
            'name':name.value
        });
            // Call the function recursively to continuously capture and send frames
            requestAnimationFrame(captureAndSend);
        }

        // Start capturing and sending frames when the video is playing
        video.onplay = () => {
            // right()
            // right()
            // up()
            captureAndSend();
        };
    } else {
        console.error("getUserMedia is not supported in this browser");
    }
});


function right() {
    const gif = document.getElementById("gif-overlay");
    if (gif.style.display == 'none' || gif.style.transform == "scaleX(-1)") {
        gif.style.transform = "scaleX(1)";
        gif.style.display = "block";
    } else {
        gif.style.display = "none";
    }
}

function left() {
    const gif = document.getElementById("gif-overlay");
    if (gif.style.display == 'none' || gif.style.transform == "scaleX(1)") {
        gif.style.display = "block";
        gif.style.transform = "scaleX(-1)";
    } else {
        gif.style.display = "none";
    }
}
function up() {
    const gif = document.getElementById("gif-overlay");
    if (gif.style.display == 'none' || gif.style.transform == "scaleY(1)") {
        gif.style.display = "block";
        gif.style.transform = "scaleY(-1)";
    } else {
        gif.style.display = "none";
    }
}
function down() {
    const gif = document.getElementById("gif-overlay");
    if (gif.style.display == 'none' || gif.style.transform == "scaleY(-1)") {
        gif.style.display = "block";
        gif.style.transform = "scaleY(-1)";
    } else {
        gif.style.display = "none";
    }
}