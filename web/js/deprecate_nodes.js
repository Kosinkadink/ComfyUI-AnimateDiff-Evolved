import { app } from '../../../scripts/app.js'

const deprecate_nodes = {
    name: 'AnimateDiff.deprecate_nodes',
    async getCustomWidgets() {
        return {
            ADEWARN(node, inputName, inputData) {
                let w = {
                    name : inputName,
                    type : "ADE.WARN",
                    value : "",
                    draw : function(ctx, node, widget_width, y, H) {
                        var show_text = app.canvas.ds.scale > 0.5;
                        var margin = 15;
                        var text_color = inputData[1]['color'] || "#FCC"
                        ctx.textAlign = "center";
                        if (show_text) {
                            if(!this.disabled)
                                ctx.stroke();
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(margin, y, widget_width - margin * 2, H);
                            ctx.clip();
                            ctx.fillStyle = text_color;
                            let disp_text = inputData[1]['text']
                            ctx.fillText(disp_text, widget_width/2, y + H * 0.7);
                            ctx.restore();
                        }

                    },
                    options : {"serialize": false},
                    computeSize : function(width) {
                        if (inputData[1]['text']) {
                            return [width, 20]
                        }
                        return [0, -4]

                    }
                }
                if (!node.widgets) {
                    node.widgets = []
                }
                node.widgets.push(w)
                return w
            }
        }
    }
}

app.registerExtension(deprecate_nodes)

