(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[83,858,485,375,69,551,602,807,26,91,571,821,502,915,619,135,503,726,931],{9093:function(e,t,l){Promise.resolve().then(l.t.bind(l,2972,23)),Promise.resolve().then(l.bind(l,2414)),Promise.resolve().then(l.bind(l,8758)),Promise.resolve().then(l.bind(l,3312)),Promise.resolve().then(l.bind(l,3362))},2414:function(e,t,l){"use strict";l.d(t,{DocsHeader:function(){return n}});var i=l(7437),a=l(9376),s=l(6994);function n(e){let{title:t}=e,l=(0,a.usePathname)(),n=s.G.find(e=>e.links.find(e=>e.href===l));return t||n?(0,i.jsxs)("header",{className:"mb-9 space-y-1",children:[n&&(0,i.jsx)("p",{className:"font-display text-sm font-medium text-sky-500",children:n.title}),t&&(0,i.jsx)("h1",{className:"font-display text-3xl tracking-tight text-slate-900 dark:text-white",children:t})]}):null}},8758:function(e,t,l){"use strict";l.d(t,{Fence:function(){return n}});var i=l(7437),a=l(2265),s=l(3331);function n(e){let{children:t="",language:l}=e;return(0,i.jsx)(s.y$,{code:t.trimEnd(),language:l,theme:{plain:{},styles:[]},children:e=>{let{className:t,style:l,tokens:s,getTokenProps:n}=e;return(0,i.jsx)("pre",{className:t,style:l,children:(0,i.jsx)("code",{children:s.map((e,t)=>(0,i.jsxs)(a.Fragment,{children:[e.filter(e=>!e.empty).map((e,t)=>(0,i.jsx)("span",{...n({token:e})},t)),"\n"]},t))})})}})}},3312:function(e,t,l){"use strict";l.d(t,{PrevNextLinks:function(){return c}});var i=l(7437),a=l(7648),s=l(9376),n=l(1994),r=l(6994);function d(e){return(0,i.jsx)("svg",{viewBox:"0 0 16 16","aria-hidden":"true",...e,children:(0,i.jsx)("path",{d:"m9.182 13.423-1.17-1.16 3.505-3.505H3V7.065h8.517l-3.506-3.5L9.181 2.4l5.512 5.511-5.511 5.512Z"})})}function o(e){let{title:t,href:l,dir:s="next",...r}=e;return(0,i.jsxs)("div",{...r,children:[(0,i.jsx)("dt",{className:"font-display text-sm font-medium text-slate-900 dark:text-white",children:"next"===s?"Next":"Previous"}),(0,i.jsx)("dd",{className:"mt-1",children:(0,i.jsxs)(a.default,{href:l,className:(0,n.Z)("flex items-center gap-x-1 text-base font-semibold text-slate-500 hover:text-slate-600 dark:text-slate-400 dark:hover:text-slate-300","previous"===s&&"flex-row-reverse"),children:[t,(0,i.jsx)(d,{className:(0,n.Z)("h-4 w-4 flex-none fill-current","previous"===s&&"-scale-x-100")})]})})]})}function c(){let e=(0,s.usePathname)(),t=r.G.flatMap(e=>e.links),l=t.findIndex(t=>t.href===e),a=l>-1?t[l-1]:null,n=l>-1?t[l+1]:null;return n||a?(0,i.jsxs)("dl",{className:"mt-12 flex border-t border-slate-200 pt-6 dark:border-slate-800",children:[a&&(0,i.jsx)(o,{dir:"previous",...a}),n&&(0,i.jsx)(o,{className:"ml-auto text-right",...n})]}):null}},3362:function(e,t,l){"use strict";l.d(t,{TableOfContents:function(){return r}});var i=l(7437),a=l(2265),s=l(7648),n=l(1994);function r(e){let{tableOfContents:t}=e,[l,r]=(0,a.useState)(t[0]?.id),d=(0,a.useCallback)(e=>e.flatMap(e=>[e.id,...e.children.map(e=>e.id)]).map(e=>{let t=document.getElementById(e);if(!t)return null;let l=parseFloat(window.getComputedStyle(t).scrollMarginTop);return{id:e,top:window.scrollY+t.getBoundingClientRect().top-l}}).filter(e=>null!==e),[]);function o(e){return e.id===l||!!e.children&&e.children.findIndex(o)>-1}return(0,a.useEffect)(()=>{if(0===t.length)return;let e=d(t);function l(){let t=window.scrollY,l=e[0].id;for(let i of e)if(t>=i.top-10)l=i.id;else break;r(l)}return window.addEventListener("scroll",l,{passive:!0}),l(),()=>{window.removeEventListener("scroll",l)}},[d,t]),(0,i.jsx)("div",{className:"hidden xl:sticky xl:top-[4.75rem] xl:-mr-6 xl:block xl:h-[calc(100vh-4.75rem)] xl:flex-none xl:overflow-y-auto xl:py-16 xl:pr-6",children:(0,i.jsx)("nav",{"aria-labelledby":"on-this-page-title",className:"w-56",children:t.length>0&&(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)("h2",{id:"on-this-page-title",className:"font-display text-sm font-medium text-slate-900 dark:text-white",children:"On this page"}),(0,i.jsx)("ol",{role:"list",className:"mt-4 space-y-3 text-sm",children:t.map(e=>(0,i.jsxs)("li",{children:[(0,i.jsx)("h3",{children:(0,i.jsx)(s.default,{href:`#${e.id}`,className:(0,n.Z)(o(e)?"text-sky-500":"font-normal text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300"),children:e.title})}),e.children.length>0&&(0,i.jsx)("ol",{role:"list",className:"mt-2 space-y-3 pl-5 text-slate-500 dark:text-slate-400",children:e.children.map(e=>(0,i.jsx)("li",{children:(0,i.jsx)(s.default,{href:`#${e.id}`,className:o(e)?"text-sky-500":"hover:text-slate-600 dark:hover:text-slate-300",children:e.title})},e.id))})]},e.id))})]})})})}},6994:function(e,t,l){"use strict";l.d(t,{G:function(){return i}});let i=[{title:"Introduction",links:[{title:"Getting started",href:"/docs/Fundamental/introduction"}]},{title:"Fundamentals",links:[{title:"Medical Imaging Formats",href:"/docs/Fundamental/image_format"},{title:"Introduction of CVD",href:"/docs/Fundamental/common_disease"},{title:"Terminologies in CVD",href:"/docs/Fundamental/terminology"}]},{title:"Data Field 6025: ECG during Fitness Test",links:[{title:"Evaluate Aortic Structure",href:"/docs/ECG_6025_20205/ecg_during_exercise"}]},{title:"Data Field 20205: ECG at Rest",links:[{title:"Evaluate Aortic Structure",href:"/docs/ECG_6025_20205/ecg_at_rest"}]},{title:"Data Field 20207: Scout Images",links:[{title:"Evaluate Aortic Structure",href:"/docs/Scout_20207/eval_aortic_structure"}]},{title:"Data Field 20208: Long Axis",links:[{title:"Evaluate Atrial Volume",href:"/docs/Long_Axis_20208/eval_atrial_volume"},{title:"Evaluate Longitudinal Strain",href:"/docs/Long_Axis_20208/eval_strain_lax"}]},{title:"Data Field 20209: Short Axis",links:[{title:"Evaluate Ventricular Volume",href:"/docs/Short_Axis_20209/eval_ventricular_volume"},{title:"Evaluate Wall Thickness",href:"/docs/Short_Axis_20209/eval_wall_thickness"},{title:"Evaluate Circumferential and Radial Strain",href:"/docs/Short_Axis_20209/eval_strain_sax"}]},{title:"Data Field 20210: Aortic Distensibilty Images",links:[{title:"Evaluate Aortic Structure",href:"/docs/Distensibility_20210/eval_aortic_dist"}]},{title:"Data Field 20211: Cine Tagging Images",links:[{title:"Evaluate Strain using Tagged MRI",href:"/docs/Tagging_20211/eval_strain_tagged"}]},{title:"Data Field 20212: LVOT Images",links:[{title:"Evaluate Left Ventricular Outflow Tract Images",href:"/docs/LVOT_20212/eval_LVOT"}]},{title:"Data Field 20213: Blood Flow Images",links:[{title:"Evaluate Phase Contrast Images",href:"/docs/Phase_Contrast_20213/eval_phase_contrast"}]},{title:"Data Field 20214: shMOLLI Sequence Images",links:[{title:"Evaluate Native T1 Images",href:"/docs/Native_T1_20214/eval_native_t1"}]}]}},function(e){e.O(0,[972,59,971,117,744],function(){return e(e.s=9093)}),_N_E=e.O()}]);