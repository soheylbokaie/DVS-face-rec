document.addEventListener("DOMContentLoaded", function() {
    // Get the video element
    const video = document.getElementById("sign-up-video");

    // Check if the browser supports getUserMedia

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