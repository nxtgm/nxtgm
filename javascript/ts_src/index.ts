
// import the pure javascript module "nxtgm_runtime_js.js"



import nxtgm_create_module from './nxtgm_javascript_runtime.js';

const NXTGM_PLUGIN_PATH = '/plugins';
const options = {env : {NXTGM_PLUGIN_PATH : NXTGM_PLUGIN_PATH}};

console.log("start create_module: ");
let _nxtgm_promise  : Promise<any> = nxtgm_create_module(options);

// prinnt "Done" when  the promise is resolved
_nxtgm_promise.then( (module) => {
    console.log("Done: creating module");
}).catch( (e) => {
    console.log("Error: ", e);
});



export async function create_module(): Promise<any>
{
    try {
        return await _nxtgm_promise;
    }
    catch (e) {
        console.log("create_module error: ", e);
    }

}


export function sum(a: number, b: number) {
    return a + b + 1;
}
