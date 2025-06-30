import { Injectable } from '@nestjs/common';
import Redis from 'ioredis';

@Injectable()
export class RedisService {
  private client = new Redis({
    host: 'redis',
    port: process.env.REDIS_URL ? parseInt(process.env.REDIS_URL) : 6379,
  });

  async setJSON(key: string, value: any) {
    await this.client.set(key, JSON.stringify(value));
  }

  async getJSON(key: string) {
    const value = await this.client.get(key);
    return value ? JSON.parse(value) : null;
  }

  async subscribe(channel: string, callback: (msg: string) => void) {
    const sub = new Redis({
      host: 'redis',
      port: process.env.REDIS_URL ? parseInt(process.env.REDIS_URL) : 6379,
    });
    await sub.subscribe(channel);
    sub.on('message', (_, msg) => callback(msg));
  }
}
