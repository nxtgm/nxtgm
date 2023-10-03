importScripts("nxtgm_javascript_runtime.js");

var nxtgm_promise = null;
var nxtgm = null;
async function main()
{
    const NXTGM_PLUGIN_PATH = '/plugins';
    var options = {env : {NXTGM_PLUGIN_PATH : NXTGM_PLUGIN_PATH}};

    let p = nxtgm_create_module(options);
    nxtgm = await p;

    // download solvers
    nxtgm_promise = nxtgm._fecht_plugin("discrete_gm_optimizer", "belief_propagation");
    await nxtgm_promise;
    console.log("nxtgm_promise", nxtgm_promise);
}

main();

async function onmessage(e)
{
    console.log("Message received from main script", e);
    console.log("awaiting nxtgm");
    await nxtgm_promise;
    console.log("nxtgm ready");
    postMessage("message from worker");
};

self.addEventListener("message", onmessage);
