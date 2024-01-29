import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

import { AppComponent } from './app.component';
import { UsuarioComponent } from './usuario/usuario.component';
import { ListaRecomendacionesComponent } from './lista-recomendaciones/lista-recomendaciones.component';

import { APP_ROUTING } from './app.routes';
import { LoginComponent } from './login/login.component';
import { NavbarComponent } from './navbar/navbar.component';
import { UsuarioService } from './services/usuario.service';


@NgModule({
  declarations: [
    AppComponent,
    UsuarioComponent,
    ListaRecomendacionesComponent,
    LoginComponent,
    NavbarComponent,
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    APP_ROUTING, 
    FormsModule
  ],
  providers: [UsuarioService],
  bootstrap: [AppComponent]
})
export class AppModule { }
