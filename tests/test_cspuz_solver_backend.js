const assert = require("assert");

require("./cspuz_solver_backend").default().then((module) => {
  const url = "https://puzz.link/p?nurikabe/6/6/m8n8i9u";
  const urlEncoded = new TextEncoder().encode(url);
  const buf = module._malloc(urlEncoded.length);
  module.HEAPU8.set(urlEncoded, buf);

  const ans = module._solve_problem(buf, urlEncoded.length);
  module._free(buf);

  const length = module.HEAPU8[ans] | (module.HEAPU8[ans + 1] << 8) | (module.HEAPU8[ans + 2] << 16) | (module.HEAPU8[ans + 3] << 24);
  const actualStr = new TextDecoder().decode(module.HEAPU8.slice(ans + 4, ans + 4 + length));
  const actual = JSON.parse(actualStr);

  const expectedStr = '{"status":"ok","description":{"kind":"grid","height":6,"width":6,"defaultStyle":"grid","data":[{"y":1,"x":1,"color":"green","item":"dot"},{"y":1,"x":3,"color":"green","item":"dot"},{"y":1,"x":7,"color":"green","item":"dot"},{"y":1,"x":9,"color":"green","item":"dot"},{"y":3,"x":3,"color":"black","item":{"kind":"text","data":"8"}},{"y":5,"x":3,"color":"green","item":"block"},{"y":5,"x":5,"color":"green","item":"block"},{"y":5,"x":7,"color":"green","item":"block"},{"y":5,"x":9,"color":"black","item":{"kind":"text","data":"8"}},{"y":5,"x":11,"color":"green","item":"dot"},{"y":7,"x":1,"color":"green","item":"dot"},{"y":7,"x":5,"color":"black","item":{"kind":"text","data":"9"}},{"y":7,"x":7,"color":"green","item":"block"},{"y":9,"x":1,"color":"green","item":"dot"},{"y":9,"x":5,"color":"green","item":"dot"},{"y":9,"x":9,"color":"green","item":"dot"},{"y":9,"x":11,"color":"green","item":"dot"},{"y":11,"x":1,"color":"green","item":"dot"},{"y":11,"x":5,"color":"green","item":"dot"},{"y":11,"x":9,"color":"green","item":"dot"},{"y":11,"x":11,"color":"green","item":"dot"}],"isUnique":false}}';
  const expected = JSON.parse(expectedStr);

  assert.deepStrictEqual(actual, expected);
});
