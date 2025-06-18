import {
  ArgumentsHost,
  Catch,
  ExceptionFilter,
  HttpException,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import {
  PrismaClientInitializationError,
  PrismaClientKnownRequestError,
  PrismaClientUnknownRequestError,
  PrismaClientValidationError,
} from '@prisma/client/runtime/library';
import { Request, Response } from 'express';

const formatRed = (msg: string) => `\x1b[31m${msg}\x1b[0m`;
const formatYellow = (msg: string) => `\x1b[33m${msg}\x1b[0m`;

@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  private readonly logger = new Logger('EXCEPTION');

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

    let status = HttpStatus.INTERNAL_SERVER_ERROR;
    let message = 'Internal server error';
    let code: HttpStatus | string = status;

    if (exception instanceof HttpException) {
      status = exception.getStatus();
      const res = exception.getResponse();
      message =
        typeof res === 'string' ? res : (res as any)?.message || message;
      code = status;
    } else if (exception instanceof PrismaClientKnownRequestError) {
      status = HttpStatus.BAD_REQUEST;
      code = exception.code;
      message = this.mapPrismaError(exception);
    } else if (
      exception instanceof PrismaClientUnknownRequestError ||
      exception instanceof PrismaClientValidationError ||
      exception instanceof PrismaClientInitializationError
    ) {
      status = HttpStatus.BAD_REQUEST;
      code = 'PRISMA_ERROR';
      message = exception.message;
    }

    const logMessage = `${request.method} ${request.url} ${status}`;

    if (status === 404) {
      this.logger.warn(formatYellow(`${logMessage} NOT FOUND`));
    } else if (status >= 500) {
      this.logger.error(
        formatRed(`${logMessage} SERVER ERROR`),
        (exception as any)?.stack,
      );
    } else {
      this.logger.warn(formatRed(`${logMessage} CLIENT ERROR)`));
    }

    const errorResponse = {
      status: 'error',
      code,
      message,
      error: process.env.NODE_ENV !== 'production' ? exception : undefined,
      path: request.url,
      timestamp: new Date().toISOString(),
    };

    response.status(status).json(errorResponse);
  }

  private mapPrismaError(exception: PrismaClientKnownRequestError): string {
    switch (exception.code) {
      case 'P2002':
        return `Campo único já está em uso: ${exception.meta?.target}`;
      case 'P2025':
        return `Registro não encontrado: ${exception.meta?.cause}`;
      default:
        return exception.message;
    }
  }
}
