import { app } from '../../../scripts/app.js'

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}
var helpDOM;
function initHelpDOM() {
    let parentDOM = document.createElement("div");
    document.body.appendChild(parentDOM)
    parentDOM.appendChild(helpDOM)
    helpDOM.className = "litegraph";
    let scrollbarStyle = document.createElement('style');
    parentDOM.className = "VHS_floatinghelp"
    scrollbarStyle.innerHTML = `
            .VHS_floatinghelp {
                scrollbar-width: 6px;
                scrollbar-color: #0003  #0000;
                &::-webkit-scrollbar {
                    background: transparent;
                    width: 6px;
                }
                &::-webkit-scrollbar-thumb {
                    background: #0005;
                    border-radius: 20px
                }
                &::-webkit-scrollbar-button {
                    display: none;
                }
            }
            .VHS_loopedvideo::-webkit-media-controls-mute-button {
                display:none;
            }
            .VHS_loopedvideo::-webkit-media-controls-fullscreen-button {
                display:none;
            }
    `
    parentDOM.appendChild(scrollbarStyle)
    chainCallback(app.canvas, "onDrawForeground", function (ctx, visible_rect){
        let n = helpDOM.node
        if (!n || !n?.graph) {
            parentDOM.style['left'] = '-5000px'
            return
        }
        //draw : function(ctx, node, widgetWidth, widgetY, height) {
        //update widget position, even if off screen
        const transform = ctx.getTransform();
        const scale = app.canvas.ds.scale;//gets the litegraph zoom
        //calculate coordinates with account for browser zoom
        const bcr = app.canvas.canvas.getBoundingClientRect()
        const x = transform.e*scale/transform.a + bcr.x;
        const y = transform.f*scale/transform.a + bcr.y;
        //TODO: text reflows at low zoom. investigate alternatives
        Object.assign(parentDOM.style, {
            left: (x+(n.pos[0] + n.size[0]+15)*scale) + "px",
            top: (y+(n.pos[1]-LiteGraph.NODE_TITLE_HEIGHT)*scale) + "px",
            width: "400px",
            minHeight: "100px",
            maxHeight: "600px",
            overflowY: 'scroll',
            transformOrigin: '0 0',
            transform: 'scale(' + scale + ',' + scale +')',
            fontSize: '18px',
            backgroundColor: LiteGraph.NODE_DEFAULT_BGCOLOR,
            boxShadow: '0 0 10px black',
            borderRadius: '4px',
            padding: '3px',
            zIndex: 3,
            position: "absolute",
            display: 'inline',
        });
    });
    function setCollapse(el, doCollapse) {
        if (doCollapse) {
            el.children[0].children[0].innerHTML = '+'
            Object.assign(el.children[1].style, {
                color: '#CCC',
                overflowX: 'hidden',
                width: '0px',
                minWidth: 'calc(100% - 20px)',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
            })
            for (let child of el.children[1].children) {
                if (child.style.display != 'none'){
                    child.origDisplay = child.style.display
                }
                child.style.display = 'none'
            }
        } else {
            el.children[0].children[0].innerHTML = '-'
            Object.assign(el.children[1].style, {
                color: '',
                overflowX: '',
                width: '100%',
                minWidth: '',
                textOverflow: '',
                whiteSpace: '',
            })
            for (let child of el.children[1].children) {
                child.style.display = child.origDisplay
            }
        }
    }
    helpDOM.collapseOnClick = function() {
        let doCollapse = this.children[0].innerHTML == '-'
        setCollapse(this.parentElement, doCollapse)
    }
    helpDOM.selectHelp = function(name, value) {
        //attempt to navigate to name in help
        function collapseUnlessMatch(items,t) {
            var match = items.querySelector('[vhs_title="' + t + '"]')
            if (!match) {
                for (let i of items.children) {
                    if (i.innerHTML.slice(0,t.length+5).includes(t)) {
                        match = i
                        break
                    }
                }
            }
            if (!match) {
                return null
            }
            //For longer documentation items with fewer collapsable elements,
            //scroll to make sure the entirety of the selected item is visible
            //This has the unfortunate side effect of trying to scroll the main
            //window if the documentation windows is forcibly offscreen,
            //but it's easy to simply scroll the main window back and seems to
            //have no visual side effects
            match.scrollIntoView(false)
            window.scrollTo(0,0)
            for (let i of items.querySelectorAll('.VHS_collapse')) {
                if (i.contains(match)) {
                    setCollapse(i, false)
                } else {
                    setCollapse(i, true)
                }
            }
            return match
        }
        let target = collapseUnlessMatch(helpDOM, name)
        if (target && value) {
            collapseUnlessMatch(target, value)
        }
    }

    helpDOM.addHelp = function(node, nodeType, description) {
        if (!description) {
            return
        }
        //Pad computed size for the clickable question mark
        let originalComputeSize = node.computeSize
        node.computeSize = function() {
            let size = originalComputeSize.apply(this, arguments)
            if (!this.title) {
                return size
            }
            let title_width = this.title.length * 0.6 * LiteGraph.NODE_TEXT_SIZE
            size[0] = Math.max(size[0], title_width + LiteGraph.NODE_TITLE_HEIGHT)
            return size
        }

        node.description = description
        chainCallback(node, "onDrawForeground", function (ctx) {
            //draw question mark
            ctx.save()
            ctx.font = 'bold 20px Arial'
            ctx.fillText("?", this.size[0]-17, -8)
            ctx.restore()
        })
        chainCallback(node, "onMouseDown", function (e, pos, canvas) {
            //On click would be preferred, but this'll be good enough
            if (pos[1] < 0 && pos[0] + LiteGraph.NODE_TITLE_HEIGHT > this.size[0]) {
                //corner question mark clicked
                if (helpDOM.node == this) {
                    helpDOM.node = undefined
                } else {
                    helpDOM.node = this;
                    helpDOM.innerHTML = this.description || "no help provided ".repeat(20)
                    for (let e of helpDOM.querySelectorAll('.VHS_collapse')) {
                        e.children[0].onclick = helpDOM.collapseOnClick
                        e.children[0].style.cursor = 'pointer'
                    }
                    for (let e of helpDOM.querySelectorAll('.VHS_precollapse')) {
                        setCollapse(e, true)
                    }
                }
                return true
            }
        })
        let timeout = null
        chainCallback(node, "onMouseMove", function (e, pos, canvas) {
            if (timeout) {
                clearTimeout(timeout)
                timeout = null
            }
            if (helpDOM.node != this) {
                return
            }
            timeout = setTimeout(() => {
                let n = this
                if (pos[0] > 0 && pos[0] < n.size[0]
                    && pos[1] > 0 && pos[1] < n.size[1]) {
                    //TODO: provide help specific to element clicked
                    let inputRows = Math.max(n.inputs.length, n.outputs.length)
                    if (pos[1] < LiteGraph.NODE_SLOT_HEIGHT * inputRows) {
                        let row = Math.floor((pos[1] - 7) / LiteGraph.NODE_SLOT_HEIGHT)
                        if (pos[0] < n.size[0]/2) {
                            if (row < n.inputs.length) {
                                helpDOM.selectHelp(n.inputs[row].name)
                            }
                        } else {
                            if (row < n.outputs.length) {
                                helpDOM.selectHelp(n.outputs[row].name)
                            }
                        }
                    } else {
                        //probably widget, but widgets have variable height.
                        let basey = LiteGraph.NODE_SLOT_HEIGHT * inputRows + 6
                        for (let w of n.widgets) {
                            if (w.y) {
                                basey = w.y
                            }
                            let wheight = LiteGraph.NODE_WIDGET_HEIGHT+4
                            if (w.computeSize) {
                                wheight = w.computeSize(n.size[0])[1]
                            }
                            if (pos[1] < basey + wheight) {
                                helpDOM.selectHelp(w.name, w.value)
                                break
                            }
                            basey += wheight
                        }
                    }
                }
            }, 500)
        })
        chainCallback(node, "onMouseLeave", function (e, pos, canvas) {
            if (timeout) {
                clearTimeout(timeout)
                timeout = null
            }
        });
    }
}



app.registerExtension({
    name: "AnimateDiffEvolved.documentation",
    async init() {
        if (app.VHSHelp) {
            helpDOM = app.VHSHelp
        } else {
            helpDOM = document.createElement("div");
            initHelpDOM()
            app.VHSHelp = helpDOM
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // NOTE: May need manual adjusting for the few non-namespaced nodes
        if(nodeData?.name?.startsWith("ADE_") && nodeData.description) {
            let description = nodeData.description
            let el = document.createElement("div")
            el.innerHTML = description
            if (!el.children.length) {
                //Is plaintext. Do minor convenience formatting
                let chunks = description.split('\n')
                nodeData.description = chunks[0]
                description = chunks.join('<br>')
            } else {
                nodeData.description = el.querySelector('#VHS_shortdesc')?.innerHTML || el.children[1]?.firstChild?.innerHTML
            }
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                helpDOM.addHelp(this, nodeType, description)
            })
        }
    },
});
