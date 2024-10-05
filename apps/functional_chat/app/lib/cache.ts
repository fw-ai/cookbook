// lib/cache.ts
import { LRUCache } from 'lru-cache';

const DEFAULT_TTL = process.env.DEFAULT_CACHE_TTL_SEC ? parseInt(process.env.DEFAULT_CACHE_TTL_SEC) : 0;

// Check if global cache exists; if not, create it
const globalCache: { cacheInstance?: LRUCache<string, any> } = global as any;
if (!globalCache.cacheInstance) {
  globalCache.cacheInstance = new LRUCache<string, any>({
    max: 100, // Maximum number of items in the cache
    ttl: DEFAULT_TTL * 1000, // Default TTL for the cache
  });
}
const cacheInstance = globalCache.cacheInstance;

export function getCache(key: string): any | undefined {
  return cacheInstance.get(key);
}

export function setCache(key: string, value: any, ttl: number): void {
  cacheInstance.set(key, value, { ttl: ttl * 1000 }); // Convert seconds to milliseconds
}

interface CacheOptions {
  keyGenerator: (...args: any[]) => string;
  ttl?: number;
}

export function cache({ keyGenerator, ttl = DEFAULT_TTL }: CacheOptions) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      const [req, res] = args;

      if (ttl <= 0) {
        // If TTL is less than or equal to 0, bypass cache
        return await originalMethod.apply(this, args);
      }

      const cacheKey = keyGenerator(...args);
      const cachedData = getCache(cacheKey);

      if (cachedData) {
        res.json(cachedData); // Set the cached response
        return;
      }

      // Capture the original send method to extract data
      const originalSend = res.json.bind(res);
      res.json = (body: any) => {
        setCache(cacheKey, body, ttl);
        originalSend(body);
      };

      await originalMethod.apply(this, args);
    };

    return descriptor;
  };
}