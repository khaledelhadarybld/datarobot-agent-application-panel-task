import { test, expect } from '@playwright/test';
import {baseURL} from "../playwright.config";
const TIMEOUT = 5 * 60 * 1000;

test.describe('Main flow test', () => {
    let context;
    let page;

    const storageStatePath = './e2e/storageState.json';

    test.beforeAll(async ({browser}) => {
        context = await browser.newContext({
            serviceWorkers: 'block',
            storageState: storageStatePath,
        });
        page = await context.newPage();
        await page.goto(baseURL, {waitUntil: 'networkidle'});
    });

    test.afterAll(async () => {
        await context?.close();
    });

    test('start new chat', async () => {
        await page.getByTestId('start-new-chat-btn').click();
        await expect(page.getByText('Assistant')).toBeVisible();
    });

    test('write a question to agent and wait until response fully done', async () => {
        test.setTimeout(TIMEOUT); // to wait long response from agent

        await page.getByTestId('start-new-chat-btn').click();
        await page.getByRole('textbox').fill('tell me small story');

        await page.getByTestId('send-message-btn').click();
        const spinner = await page.getByTestId('thinking-loading')
        await expect(spinner).toBeVisible();
        await expect(page.getByTestId('send-message-disabled-btn')).toBeDisabled()

        // wait until message will be received
        await page.getByTestId('send-message-btn').waitFor({ state: 'visible', timeout: TIMEOUT });
    });

    test('new thread is active when user switches chats', async () => {
        test.setTimeout(TIMEOUT); // to wait for chat to be deleted
        let contentCount = 0
        await page.getByTestId('start-new-chat-btn').click();

        await page.getByRole('textbox').fill('tell some fun fact');
        await page.getByTestId('send-message-btn').click();

        await page.locator('#sidebar-chats').locator('[test-id^="chat-"]')?.last().click();
        await expect(page.getByTestId('send-message-btn')).toBeVisible();

        await page.locator('#sidebar-chats').locator('[test-id^="chat-"]')?.first().click();

        await expect.poll(
            async () => {
                const content = await page.locator('[test-id^="default-assistant-message-"]')?.first();
                const text = await content.textContent();
                contentCount =  text?.length ?? 0;
                return contentCount;
            },
            {
                timeout: TIMEOUT,
            }
        ).toBeGreaterThan(0);
    })

    test('remove chat', async () => {
        test.setTimeout(TIMEOUT); // to wait for chat to be deleted
        await page.locator('#sidebar-chats').locator('[test-id^="chat-"]')?.last().click();
        const oldUrl = page.url();
        await page.locator('.dropdown-menu-trigger')?.last().click();
        await page.getByTestId('delete-chat-menu-item').click();
        await page.getByTestId('modal-confirm').click();
        await expect.poll(() => page.url(), {
            timeout: TIMEOUT,
        }).not.toBe(oldUrl); // check if user was redirected to the active chat
    })
})
