(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[924,321,559,807,135,250,494,681,215,850,698,926,30,864,193,18,840,43,682,882,3,683,208,931,605],{9093:function(e,t,l){Promise.resolve().then(l.t.bind(l,2972,23)),Promise.resolve().then(l.bind(l,2414)),Promise.resolve().then(l.bind(l,8758)),Promise.resolve().then(l.bind(l,3312)),Promise.resolve().then(l.bind(l,3362))},2414:function(e,t,l){"use strict";l.d(t,{DocsHeader:function(){return i}});var n=l(7437),s=l(9376),r=l(6994);function i(e){let{title:t}=e,l=(0,s.usePathname)(),i=r.G.find(e=>e.links.find(e=>e.href===l));return t||i?(0,n.jsxs)("header",{className:"mb-9 space-y-1",children:[i&&(0,n.jsx)("p",{className:"font-display text-sm font-medium text-sky-500",children:i.title}),t&&(0,n.jsx)("h1",{className:"font-display text-3xl tracking-tight text-slate-900 dark:text-white",children:t})]}):null}},8758:function(e,t,l){"use strict";l.d(t,{Fence:function(){return i}});var n=l(7437),s=l(2265),r=l(3331);function i(e){let{children:t,language:l}=e;return(0,n.jsx)(r.y$,{code:t.trimEnd(),language:l,theme:{plain:{},styles:[]},children:e=>{let{className:t,style:l,tokens:r,getTokenProps:i}=e;return(0,n.jsx)("pre",{className:t,style:l,children:(0,n.jsx)("code",{children:r.map((e,t)=>(0,n.jsxs)(s.Fragment,{children:[e.filter(e=>!e.empty).map((e,t)=>(0,n.jsx)("span",{...i({token:e})},t)),"\n"]},t))})})}})}},3312:function(e,t,l){"use strict";l.d(t,{PrevNextLinks:function(){return c}});var n=l(7437),s=l(7648),r=l(9376),i=l(1994),a=l(6994);function d(e){return(0,n.jsx)("svg",{viewBox:"0 0 16 16","aria-hidden":"true",...e,children:(0,n.jsx)("path",{d:"m9.182 13.423-1.17-1.16 3.505-3.505H3V7.065h8.517l-3.506-3.5L9.181 2.4l5.512 5.511-5.511 5.512Z"})})}function o(e){let{title:t,href:l,dir:r="next",...a}=e;return(0,n.jsxs)("div",{...a,children:[(0,n.jsx)("dt",{className:"font-display text-sm font-medium text-slate-900 dark:text-white",children:"next"===r?"Next":"Previous"}),(0,n.jsx)("dd",{className:"mt-1",children:(0,n.jsxs)(s.default,{href:l,className:(0,i.Z)("flex items-center gap-x-1 text-base font-semibold text-slate-500 hover:text-slate-600 dark:text-slate-400 dark:hover:text-slate-300","previous"===r&&"flex-row-reverse"),children:[t,(0,n.jsx)(d,{className:(0,i.Z)("h-4 w-4 flex-none fill-current","previous"===r&&"-scale-x-100")})]})})]})}function c(){let e=(0,r.usePathname)(),t=a.G.flatMap(e=>e.links),l=t.findIndex(t=>t.href===e),s=l>-1?t[l-1]:null,i=l>-1?t[l+1]:null;return i||s?(0,n.jsxs)("dl",{className:"mt-12 flex border-t border-slate-200 pt-6 dark:border-slate-800",children:[s&&(0,n.jsx)(o,{dir:"previous",...s}),i&&(0,n.jsx)(o,{className:"ml-auto text-right",...i})]}):null}},3362:function(e,t,l){"use strict";l.d(t,{TableOfContents:function(){return a}});var n=l(7437),s=l(2265),r=l(7648),i=l(1994);function a(e){let{tableOfContents:t}=e,[l,a]=(0,s.useState)(t[0]?.id),d=(0,s.useCallback)(e=>e.flatMap(e=>[e.id,...e.children.map(e=>e.id)]).map(e=>{let t=document.getElementById(e);if(!t)return null;let l=parseFloat(window.getComputedStyle(t).scrollMarginTop);return{id:e,top:window.scrollY+t.getBoundingClientRect().top-l}}).filter(e=>null!==e),[]);function o(e){return e.id===l||!!e.children&&e.children.findIndex(o)>-1}return(0,s.useEffect)(()=>{if(0===t.length)return;let e=d(t);function l(){let t=window.scrollY,l=e[0].id;for(let n of e)if(t>=n.top-10)l=n.id;else break;a(l)}return window.addEventListener("scroll",l,{passive:!0}),l(),()=>{window.removeEventListener("scroll",l)}},[d,t]),(0,n.jsx)("div",{className:"hidden xl:sticky xl:top-[4.75rem] xl:-mr-6 xl:block xl:h-[calc(100vh-4.75rem)] xl:flex-none xl:overflow-y-auto xl:py-16 xl:pr-6",children:(0,n.jsx)("nav",{"aria-labelledby":"on-this-page-title",className:"w-56",children:t.length>0&&(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)("h2",{id:"on-this-page-title",className:"font-display text-sm font-medium text-slate-900 dark:text-white",children:"On this page"}),(0,n.jsx)("ol",{role:"list",className:"mt-4 space-y-3 text-sm",children:t.map(e=>(0,n.jsxs)("li",{children:[(0,n.jsx)("h3",{children:(0,n.jsx)(r.default,{href:`#${e.id}`,className:(0,i.Z)(o(e)?"text-sky-500":"font-normal text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300"),children:e.title})}),e.children.length>0&&(0,n.jsx)("ol",{role:"list",className:"mt-2 space-y-3 pl-5 text-slate-500 dark:text-slate-400",children:e.children.map(e=>(0,n.jsx)("li",{children:(0,n.jsx)(r.default,{href:`#${e.id}`,className:o(e)?"text-sky-500":"hover:text-slate-600 dark:hover:text-slate-300",children:e.title})},e.id))})]},e.id))})]})})})}},6994:function(e,t,l){"use strict";l.d(t,{G:function(){return n}});let n=[{title:"Introduction",links:[{title:"Getting started",href:"/"}]},{title:"Common Knowledge",links:[{title:"Getting started",href:"/"}]},{title:"Data Field 20208: Long Axis",links:[{title:"Evaluate Atrial Volume",href:"/docs/Long_Axis_20208/eval_atrial_volume"}]},{title:"Data Field 20209: Short Axis",links:[{title:"Evaluate Ventricular Volume",href:"/docs/Short_Axis_20209/eval_ventricular_volume"}]}]}},function(e){e.O(0,[972,59,971,117,744],function(){return e(e.s=9093)}),_N_E=e.O()}]);