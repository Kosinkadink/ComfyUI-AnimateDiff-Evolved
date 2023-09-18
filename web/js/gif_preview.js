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
      .translateSelf(0, widgetY + margin)
  
    const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
    Object.assign(widget.inputEl.style, {
      transformOrigin: '0 0',
      transform: scale,
      left: `${transform.e}px`,
      top: `${transform.d + transform.f}px`,
      width: `${widgetWidth}px`,
      height: `${(height || widget.parent?.inputHeight || 32) - margin}px`,
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

const CreatePreviewElement = (name, val, format) => {
  const [type] = format.split('/')  
  const w = {
      name,
      type,
      value: val,
      draw: function (ctx, node, widgetWidth, widgetY, height) {
        const [cw, ch] = this.computeSize(widgetWidth)
        offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, ch)
      },
      computeSize: function (_) {
        const ratio = this.inputRatio || 1
        const width = Math.max(220, this.parent.size[0])
        return [width, (width / ratio + 10)]
      },
      onRemoved: function () {
        if (this.inputEl) {
          this.inputEl.remove()
        }
      },
    }

    w.inputEl = document.createElement(type === 'video' ? 'video' : 'img')
    w.inputEl.src = w.value
    if (type === 'video') {
      w.inputEl.setAttribute('type', 'video/webm');
      w.inputEl.autoplay = true
      w.inputEl.loop = true
      w.inputEl.controls = false;
    }
    w.inputEl.onload = function () {
      w.inputRatio = w.inputEl.naturalWidth / w.inputEl.naturalHeight
    }
    document.body.appendChild(w.inputEl)
    return w
  }

const gif_preview = {
    name: 'AnimateDiff.gif_preview',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        switch (nodeData.name) {
            case 'ADE_AnimateDiffCombine':{
                const onExecuted = nodeType.prototype.onExecuted
                nodeType.prototype.onExecuted = function (message) {
                const prefix = 'ad_gif_preview_'
                const r = onExecuted ? onExecuted.apply(this, message) : undefined

                if (this.widgets) {
                    const pos = this.widgets.findIndex((w) => w.name === `${prefix}_0`)
                    if (pos !== -1) {
                      for (let i = pos; i < this.widgets.length; i++) {
                        this.widgets[i].onRemoved?.()
                    }
                    this.widgets.length = pos
                    }
                    if (message?.gifs) {
                      message.gifs.forEach((params, i) => {
                        const previewUrl = api.apiURL(
                          '/view?' + new URLSearchParams(params).toString()
                        )
                  const w = this.addCustomWidget(
                    CreatePreviewElement(`${prefix}_${i}`, previewUrl, params.format || 'image/gif')
                  )
                  w.parent = this
                  })
                 }
                  const onRemoved = this.onRemoved
                  this.onRemoved = () => {
                    cleanupNode(this)
                    return onRemoved?.()
                  }
                }
                this.setSize([this.size[0], this.computeSize([this.size[0], this.size[1]])[1]])
                return r
                }
                break
              }
        }
    }
}

app.registerExtension(gif_preview)
