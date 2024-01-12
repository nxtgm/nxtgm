
// import the pure javascript module "nxtgm_runtime_js.js"



import nxtgm_create_module from './nxtgm_javascript_runtime.js';

// const NXTGM_PLUGIN_PATH = '/plugins';
// const options = {env : {NXTGM_PLUGIN_PATH : NXTGM_PLUGIN_PATH}};





async function create_module(
    options : any
): Promise<any>
{
    try {
        let _nxtgm_promise  : Promise<any> = nxtgm_create_module(options);
        return await _nxtgm_promise;
    }
    catch (e) {
        console.log("create_module error: ", e);
    }
}


export {create_module};
