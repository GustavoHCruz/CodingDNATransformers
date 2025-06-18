import {
  CallHandler,
  ExecutionContext,
  Injectable,
  Logger,
  NestInterceptor,
} from '@nestjs/common';
import { Observable, tap } from 'rxjs';

const formatBlue = (msg: string) => `\x1b[36m${msg}\x1b[0m`;

@Injectable()
export class LoggingInterceptor implements NestInterceptor {
  private readonly logger = new Logger('HTTP');

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest();

    const { method, originalUrl, body, query, params } = request;

    const now = Date.now();

    return next.handle().pipe(
      tap(() => {
        const delay = Date.now() - now;

        const isProd = process.env.NODE_ENV === 'production';

        const logMessage = `${method} ${originalUrl} ${delay}ms`;

        const paramsMessage = params
          ? ` | Params: ${JSON.stringify(params)}`
          : '';
        const queryMessage = query ? ` | Query: ${JSON.stringify(query)}` : '';
        const bodyMessage = body ? ` | Body: ${JSON.stringify(body)}` : '';

        if (isProd) {
          if (originalUrl.includes('/critical') || method === 'DELETE') {
            this.logger.warn(formatBlue(`PROD: ${logMessage}`));
          }
        } else {
          this.logger.log(
            formatBlue(
              `${logMessage}${paramsMessage}${queryMessage}${bodyMessage}`,
            ),
          );
        }
      }),
    );
  }
}
