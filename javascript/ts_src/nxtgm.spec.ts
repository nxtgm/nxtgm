import { create_module } from "./index";

beforeEach((): void => {
  jest.setTimeout(60000);
});

test('the data is peanut butter', async () => {
  await create_module();
});
