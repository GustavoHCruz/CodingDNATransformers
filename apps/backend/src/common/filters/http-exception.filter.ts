import {
  ArgumentsHost,
  Catch,
  ExceptionFilter,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import {
  PrismaClientInitializationError,
  PrismaClientKnownRequestError,
  PrismaClientUnknownRequestError,
  PrismaClientValidationError,
} from '@prisma/client/runtime/library';
import { Request, Response } from 'express';

@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
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
