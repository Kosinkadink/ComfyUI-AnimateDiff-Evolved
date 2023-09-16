import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

function offsetDOMWidget(
    widget,
    ctx,
    node,
    widgetWidth,
    widgetY,
    height
  ) {
    const margin = 10
    const elRect = ctx.canvas.getBoundingClientRect()
    const transform = new DOMMatrix()
      .scaleSelf(
        elRect.width / ctx.canvas.width,
        elRect.height / ctx.canvas.height
      )
      .multiplySelf(ctx.getTransform())
      .translateSelf(margin, margin + widgetY)
  
    const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
    Object.assign(widget.inputEl.style, {
      transformOrigin: '0 0',
      transform: scale,
      left: `${transform.a + transform.e}px`,
      top: `${transform.d + transform.f}px`,
      width: `${widgetWidth - margin * 2}px`,
      // height: `${(widget.parent?.inputHeight || 32) - (margin * 2)}px`,
      height: `${(height || widget.parent?.inputHeight || 32) - margin * 2}px`,
  
      position: 'absolute',
      background: !node.color ? '' : node.color,
      color: !node.color ? '' : 'white',
      zIndex: 5, //app.graph._nodes.indexOf(node),
    })
  }

  export const hasWidgets = (node) => {
    if (!node.widgets || !node.widgets?.[Symbol.iterator]) {
      return false
    }
    return true
  }

  export const cleanupNode = (node) => {
    if (!hasWidgets(node)) {
      return
    }
  
    for (const w of node.widgets) {
      if (w.canvas) {
        w.canvas.remove()
      }
      if (w.inputEl) {
        w.inputEl.remove()
      }
      // calls the widget remove callback
      w.onRemoved?.()
    }
  }

const CreateVideoElement = (name, val) => {
    const w = {
      name,
      type: 'video',
      value: val,
      draw: function (ctx, node, widgetWidth, widgetY, height) {
        const [cw, ch] = this.computeSize(widgetWidth)
        offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, ch)
      },
      computeSize: function (width) {
        const ratio = this.inputRatio || 1
        if (width) {
          return [width, width / ratio + 4]
        }
        return [128, 128]
      },
      onRemoved: function () {
        if (this.inputEl) {
          this.inputEl.remove()
        }
      },
    }

    w.inputEl = document.createElement('video')
    w.inputEl.src = w.value
    w.inputEl.setAttribute('type', 'video/webm');
    w.inputEl.autoplay = true
    w.inputEl.loop = true
    w.inputEl.controls = false; // Add controls to the video element
    w.inputEl.onload = function () {
      w.inputRatio = w.inputEl.naturalWidth / w.inputEl.naturalHeight
    }
    document.body.appendChild(w.inputEl)
    return w
  }

  const CreateImgElement = (name, val) => {
    const w = {
      name,
      type: 'image',
      value: val,
      draw: function (ctx, node, widgetWidth, widgetY, height) {
        const [cw, ch] = this.computeSize(widgetWidth)
        offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, ch)
      },
      computeSize: function (width) {
        const ratio = this.inputRatio || 1
        if (width) {
          return [width, width / ratio + 4]
        }
        return [128, 128]
      },
      onRemoved: function () {
        if (this.inputEl) {
          this.inputEl.remove()
        }
      },
    }

    w.inputEl = document.createElement('img')
    w.inputEl.src = w.value
    w.inputEl.controls = false; // Add controls to the video element
    w.inputEl.onload = function () {
      w.inputRatio = w.inputEl.naturalWidth / w.inputEl.naturalHeight
    }
    document.body.appendChild(w.inputEl)
    return w
  }

const gif_preview = {
    name: 'gif.preview',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        switch (nodeData.name) {
            case 'ADE_AnimateDiffCombineVideo':
            case 'ADE_AnimateDiffCombine':{
                const onExecuted = nodeType.prototype.onExecuted
                nodeType.prototype.onExecuted = function (message) {
                const prefix = 'anything_'
                const r = onExecuted ? onExecuted.apply(this, message) : undefined

                if (this.widgets) {
                    const pos = this.widgets.findIndex((w) => w.name === `${prefix}_0`)
                    if (pos !== -1) {
                    for (let i = pos; i < this.widgets.length; i++) {
                        this.widgets[i].onRemoved?.()
                    }
                    this.widgets.length = pos
                    }

                    let imgURLs = []
                    let videoURLs = []
                    if (message) {
                    if (message.gif) {
                        imgURLs = imgURLs.concat(
                        message.gif.map((params) => {
                            return api.apiURL(
                            '/view?' + new URLSearchParams(params).toString()
                            )
                        })
                        )
                    }
                    if (message.video) {
                      videoURLs = videoURLs.concat(
                      message.video.map((params) => {
                          return api.apiURL(
                          '/view?' + new URLSearchParams(params).toString()
                          )
                      })
                      )
                  }
                    let i = 0
                    for (const img of imgURLs) {
                        const w = this.addCustomWidget(
                        CreateImgElement(`${prefix}_${i}`, img)
                        )
                        w.parent = this
                        i++
                    }
                    for (const video of videoURLs) {
                      const w = this.addCustomWidget(
                      CreateVideoElement(`${prefix}_${i}`, video)
                      )
                      w.parent = this
                      i++
                  }
                    }
                    const onRemoved = this.onRemoved
                    this.onRemoved = () => {
                    cleanupNode(this)
                    return onRemoved?.()
                    }
                }
                //this.setSize?.(this.computeSize())  # this seems to reset the node size on each generation
                return r
                }

                break
        }
        }
    }
}

app.registerExtension(gif_preview)
