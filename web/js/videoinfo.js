import { app } from '../../../scripts/app.js'


function getVideoMetadata(file) {
    return new Promise((r) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const videoData = new Uint8Array(event.target.result);
            const dataView = new DataView(videoData.buffer);

            let txt = "";
            // Check for known valid magic strings
            if (dataView.getUint32(0) == 0x1A45DFA3) {
                //webm
                //see http://wiki.webmproject.org/webm-metadata/global-metadata
                //and maybe https://www.webmproject.org/docs/container/
                console.log("parsing webm");
                //contrary to specs, tag seems consistently at start
                //COMMENT + 0x4487 + packed length?
                //length 0x8d9 becomes 0x48d8
                let offset = 4 + 8; //COMMENT is 7 chars + 1 to realign
                while(offset < videoData.length) {
                    //Check for text tags
                    if (dataView.getUint16(offset) == 0x4487) {
                        //check that name of tag is COMMENT
                        const name = String.fromCharCode(...videoData.slice(offset-7,offset));
                        if (name === "COMMENT") {
                            console.log("found comment");
                            let length = dataView.getUint16(offset+2) & 0x3FFF;
                            console.log(length);
                            const content = String.fromCharCode(...videoData.slice(offset+4, offset+4+length));
                            console.log(content);
                            r(JSON.parse(content));
                            return;
                        }
                    }
                    offset+=2;
                }
            } else if (dataView.getUint32(4) == 0x66747970 && dataView.getUint32(8) == 0x69736F6D) {
                //mp4
                console.error("not yet implemented")
            } else {
                console.error("Unknown magic: " + dataView.getUint32(0))
                r();
                return;
            }

        };

        reader.readAsArrayBuffer(file);
    });
}
function isVideoFile(file) {
    if (file.name?.endsWith(".webm")) {
        return true;
    }
    if (file.name?.endsWith(".mp4")) {
        return true;
    }

    return false;
}

async function handleFile(file) {
    console.log("intercepted file call");
    if (file.type.startsWith("video/") || isVideoFile(file)) {
        console.log("got video");
        const videoInfo = await getVideoMetadata(file);
        if (videoInfo) {
            if (videoInfo.workflow) {
                console.log("loading workflow");
                app.loadGraphData(videoInfo.workflow);
            }
            //Potentially check for/parse A1111 metadata here.
        }
    } else {
        console.log("calling original HandleFile");
        await app.originalHandleFile(file);
    }
}

//Storing the original function in app is probably a major no-no
//But it's the only way I've found to maintain keep the 'this' reference
app.originalHandleFile = app.handleFile;
app.handleFile = handleFile;
//hijack comfy-file-input to allow webm/mp4
document.getElementById("comfy-file-input").accept += ",video/webm,video/mp4";
