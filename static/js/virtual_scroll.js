/**
 * 虚拟滚动模块 —— 优化大量消息时的渲染性能
 * 
 * 原理：只渲染可视区域内的消息，上下预留缓冲区域
 * 当用户滚动时，动态加载/卸载消息DOM
 * 
 * 使用场景：消息数量 > 100 条时自动启用
 */

class VirtualScroller {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) throw new Error(`Container #${containerId} not found`);
    
    // 配置
    this.itemHeight = options.itemHeight || 80;      // 预估单项高度
    this.bufferSize = options.bufferSize || 10;       // 上下缓冲数量
    this.threshold = options.threshold || 100;        // 启用阈值
    
    // 状态
    this.messages = [];        // 所有消息数据
    this.visibleItems = [];    // 当前可见项索引
    this.scrollTop = 0;
    this.containerHeight = 0;
    this.isVirtual = false;
    
    // 滚动节流
    this._scrollTimer = null;
    this._lastScrollTime = 0;
    
    this._init();
  }
  
  _init() {
    // 监听滚动
    this.container.addEventListener('scroll', () => this._onScroll(), { passive: true });
    
    // 监听容器大小变化
    this._resizeObserver = new ResizeObserver(() => {
      this.containerHeight = this.container.clientHeight;
      this._updateVisibleItems();
    });
    this._resizeObserver.observe(this.container);
    
    this.containerHeight = this.container.clientHeight;
  }
  
  /**
   * 设置消息数据
   */
  setMessages(messages) {
    this.messages = messages;
    this.isVirtual = messages.length > this.threshold;
    
    if (this.isVirtual) {
      this._enableVirtual();
    } else {
      this._disableVirtual();
    }
    
    this._updateVisibleItems();
  }
  
  /**
   * 添加单条消息
   */
  addMessage(msg) {
    this.messages.push(msg);
    
    // 如果之前不是虚拟滚动，检查是否需要切换
    if (!this.isVirtual && this.messages.length > this.threshold) {
      this.isVirtual = true;
      this._enableVirtual();
    }
    
    // 如果在底部，自动滚动
    const isAtBottom = this.container.scrollTop + this.containerHeight >= 
      this.container.scrollHeight - 100;
    
    this._updateVisibleItems();
    
    if (isAtBottom) {
      this.scrollToBottom();
    }
  }
  
  /**
   * 启用虚拟滚动模式
   */
  _enableVirtual() {
    // 创建占位容器
    if (!this.phantom) {
      this.phantom = document.createElement('div');
      this.phantom.className = 'virtual-phantom';
      this.phantom.style.cssText = 'position:relative;width:100%;';
      this.container.style.position = 'relative';
      this.container.insertBefore(this.phantom, this.container.firstChild);
    }
    
    // 将现有消息移入内容层
    this._contentLayer = document.createElement('div');
    this._contentLayer.className = 'virtual-content';
    this._contentLayer.style.cssText = 'position:absolute;top:0;left:0;width:100%;';
    this.container.appendChild(this._contentLayer);
  }
  
  /**
   * 禁用虚拟滚动（消息少时直接渲染）
   */
  _disableVirtual() {
    if (this.phantom) {
      this.phantom.style.height = '0';
    }
    // 直接渲染所有消息
    this._renderAll();
  }
  
  /**
   * 滚动事件处理（节流）
   */
  _onScroll() {
    const now = Date.now();
    if (now - this._lastScrollTime < 16) { // ~60fps
      if (this._scrollTimer) clearTimeout(this._scrollTimer);
      this._scrollTimer = setTimeout(() => this._onScroll(), 16);
      return;
    }
    this._lastScrollTime = now;
    this.scrollTop = this.container.scrollTop;
    this._updateVisibleItems();
  }
  
  /**
   * 计算并更新可见项
   */
  _updateVisibleItems() {
    if (!this.isVirtual) return;
    
    const totalHeight = this.messages.length * this.itemHeight;
    this.phantom.style.height = totalHeight + 'px';
    
    // 计算可见范围
    const startIdx = Math.max(0, Math.floor(this.scrollTop / this.itemHeight) - this.bufferSize);
    const endIdx = Math.min(
      this.messages.length,
      Math.ceil((this.scrollTop + this.containerHeight) / this.itemHeight) + this.bufferSize
    );
    
    // 只渲染可见区域
    this._renderRange(startIdx, endIdx);
  }
  
  /**
   * 渲染指定范围的消息
   */
  _renderRange(start, end) {
    if (!this._contentLayer) return;
    
    const offsetY = start * this.itemHeight;
    this._contentLayer.style.transform = `translateY(${offsetY}px)`;
    
    // 生成HTML
    const html = this.messages.slice(start, end).map((msg, i) => {
      const idx = start + i;
      return this._renderMessage(msg, idx);
    }).join('');
    
    this._contentLayer.innerHTML = html;
  }
  
  /**
   * 渲染单条消息（子类可覆盖）
   */
  _renderMessage(msg, idx) {
    // 默认实现：返回消息HTML
    return `<div class="msg-item" data-idx="${idx}" style="height:${this.itemHeight}px">
      ${msg.html || msg.content || ''}
    </div>`;
  }
  
  /**
   * 渲染所有消息（非虚拟模式）
   */
  _renderAll() {
    // 清除虚拟滚动相关元素
    if (this._contentLayer) {
      this._contentLayer.remove();
      this._contentLayer = null;
    }
    // 保留原始消息DOM
  }
  
  /**
   * 滚动到底部
   */
  scrollToBottom() {
    if (this.isVirtual) {
      this.container.scrollTop = this.messages.length * this.itemHeight;
    } else {
      this.container.scrollTop = this.container.scrollHeight;
    }
  }
  
  /**
   * 销毁
   */
  destroy() {
    if (this._scrollTimer) clearTimeout(this._scrollTimer);
    if (this._resizeObserver) this._resizeObserver.disconnect();
    if (this.phantom) this.phantom.remove();
    if (this._contentLayer) this._contentLayer.remove();
  }
}

// 导出
if (typeof module !== 'undefined' && module.exports) {
  module.exports = VirtualScroller;
}
