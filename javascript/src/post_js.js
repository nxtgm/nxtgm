addOnPreRun(function() {
    if (Module['env'] && typeof Module['env'] === 'object')
    {
        for (var key in Module['env'])
        {
            if (Module['env'].hasOwnProperty(key))
                ENV[key] = Module['env'][key];
        }
    }
});

function create_lock()
{
    let _lock = Promise.resolve();

    async function acquire_lock()
    {
        const old_lock = _lock;
        let release_lock = () => {};
        _lock = new Promise((resolve) => (release_lock = resolve));
        await old_lock;
        return release_lock;
    }
    return acquire_lock;
}

function create_dynlib_fs(lib, searchDirs)
{
    const dirname = lib.substring(0, lib.lastIndexOf("/"));
    let _searchDirs = searchDirs || [];
    _searchDirs = _searchDirs.concat([ dirname ], [ `/lib` ]);

    const resolvePath = (path) => {
        if (path.startsWith("/") && path.endsWith(".so") && Module.FS.findObject(path) !== null)
        {
            return path;
        }

        if (Module.PATH.basename(path) !== Module.PATH.basename(lib))
        {
            console.log(`Searching a library from ${path}, required by ${lib}`);
        }

        for (const dir of _searchDirs)
        {
            const fullPath = Module.PATH.join2(dir, path);
            if (Module.FS.findObject(fullPath) !== null)
            {
                return fullPath;
            }
        }
        return path;
    };

    let readFile = (path) => Module.FS.readFile(resolvePath(path));

    const fs = {
        findObject : (path, dontResolveLastLink) => {
            let obj = Module.FS.findObject(resolvePath(path), dontResolveLastLink);

            if (obj === null)
            {
                console.log(`Failed to find a library: ${resolvePath(path)}`);
            }

            return obj;
        },
        readFile : readFile,
    };

    return fs;
}

Module["__eq__"] =
    function(a, b) {
    return a === b;
}

Module['_new'] =
    function(cls, ...args) {
    return new cls(...args);
}

Module['_instanceof'] =
    function(instance, cls) {
    return (instance instanceof cls);
}

Module["_typeof"] =
    function(x) {
    return typeof x;
}

Module["_delete"] =
    function(x, key) {
    delete x[key];
}

Module["_is_double_array"] =
    function(x) {
    return x instanceof Float64Array;
}

Module["_fetched_plugins"] = {};

function try_make_dir(path)
{
    try
    {
        if (!FS.isDir(path))
        {
            FS.mkdir(path);
        }
    }
    catch (e)
    {
        // console.log("mkdir error", path, e);
    }
}

const acquireDynlibLock = create_lock();
Module["_fecht_plugin"] =
    async function(plugin_family, plugin_name, plugin_base_url) {
    if (plugin_base_url === undefined)
    {
        plugin_base_url = "./plugins";
    }

    let plugin_folder = ENV["NXTGM_PLUGIN_PATH"];

    // check cache
    if (plugin_family in Module["_fetched_plugins"])
    {
        if (plugin_name in Module["_fetched_plugins"][plugin_family])
        {
            return;
        }
    }

    let plugin_url = plugin_base_url + "/" + plugin_family + "/" +
                     "lib" + plugin_family + "_" + plugin_name + ".so";
    let result = await fetch(plugin_url);
    // check
    if (!result.ok)
    {
        throw new Error("HTTP error " + result.status);
    }

    let plugin_array_buffer = await result.arrayBuffer();
    const plugin_array = new Uint8Array(plugin_array_buffer)
    const plugin_filename = "lib" + plugin_family + "_" + plugin_name + ".so";
    const plugin_path = plugin_folder + "/" + plugin_family + "/" + plugin_filename

    try_make_dir(plugin_folder);
    try_make_dir(plugin_folder + "/" + plugin_family);
    FS.writeFile(plugin_path, plugin_array);

    const releaseDynlibLock = await acquireDynlibLock();
    try
    {

        await loadDynamicLibrary(plugin_path, {
            loadAsync : true,
            nodelete : true,
            allowUndefined : true,
            global : true,
            fs : create_dynlib_fs(plugin_path, [])
        })

        // full path to lib
        const dsoOnlyLibName = Module.LDSO.loadedLibsByName[plugin_filename];
        const dsoFullLib = Module.LDSO.loadedLibsByName[plugin_path];

        if (!dsoOnlyLibName && !dsoFullLib)
        {
            console.execption(`Failed to load ${plugin_path}: LDSO not found`);
        }
        if (!dsoOnlyLibName)
        {

            Module.LDSO.loadedLibsByName[plugin_filename] = dsoFullLib
        }

        if (!dsoFullLib)
        {
            Module.LDSO.loadedLibsByName[plugin_path] = dsoOnlyLibName;
        }

        // add to loaded plugins
        if (!(plugin_family in Module["_fetched_plugins"]))
        {
            Module["_fetched_plugins"][plugin_family] = {};
        }
        Module["_fetched_plugins"][plugin_family][plugin_name] = true;
    }
    finally
    {
        releaseDynlibLock();
    }
}

Module['status_name'] =
    function(status) {
    return status.constructor.name.slice(19);
}

    Module.FS = FS;
Module.PATH = PATH;
Module.LDSO = LDSO;
