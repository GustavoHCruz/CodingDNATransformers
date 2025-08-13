import { Observable } from 'rxjs';

export function observableToAsyncIterable<T>(
  observable: Observable<T>,
): AsyncIterable<T> {
  return {
    [Symbol.asyncIterator]() {
      const buffer: T[] = [];
      const queue: ((value: IteratorResult<T>) => void)[] = [];
      let isComplete = false;
      let hasError = false;
      let storedError: any = null;

      const subscription = observable.subscribe({
        next: (value) => {
          if (queue.length > 0) {
            const resolve = queue.shift()!;
            resolve({ value, done: false });
          } else {
            buffer.push(value);
          }
        },
        error: (err) => {
          hasError = true;
          storedError = err;
          while (queue.length > 0) {
            const resolve = queue.shift()!;
            resolve(Promise.reject(storedError) as any);
          }
        },
        complete: () => {
          isComplete = true;
          while (queue.length > 0) {
            const resolve = queue.shift()!;
            resolve({ value: undefined, done: true });
          }
        },
      });

      return {
        async next(): Promise<IteratorResult<T>> {
          if (hasError) throw storedError;
          if (buffer.length > 0) {
            return { value: buffer.shift()!, done: false };
          }
          if (isComplete) {
            return { value: undefined, done: true };
          }
          return new Promise<IteratorResult<T>>((resolve) => {
            queue.push(resolve);
          });
        },
        async return() {
          subscription.unsubscribe();
          return { value: undefined, done: true };
        },
        async throw(err) {
          subscription.unsubscribe();
          throw err;
        },
      };
    },
  };
}
