import { Component, OnInit } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { UsuarioService } from '../services/usuario.service';

@Component({
  selector: 'app-lista-recomendaciones',
  templateUrl: './lista-recomendaciones.component.html',
  styleUrls: ['./lista-recomendaciones.component.css']
})
export class ListaRecomendacionesComponent implements OnInit {

  recomendaciones : any[] = [];

  features : any[] = [];

  usersImportantes : any[] = [];

  htmlMapa : string = "";
  safeHtml: SafeHtml= "";

  constructor(private usuarioService: UsuarioService, private sanitizer: DomSanitizer) { 
    this.usuarioService.get_recomendaciones_by_id(usuarioService.idLogged).subscribe((data:any)=>{
      console.log(data);
      console.log(JSON.parse(data["recommendaciones"]));
      console.log(JSON.parse(data["usuarios"]));
      console.log(data["features"]);

      this.recomendaciones = JSON.parse(data["recommendaciones"]);
      this.usersImportantes = JSON.parse(data["usuarios"]);
      this.features = data["features"];

      


      // this.usuarioService.get_mapa(recomendaciones).subscribe((data:any) =>{
      //   console.log(data);

      //   let mapa = document.createElement('div');

      //   let index = data.indexOf("<script>")
      //   let data2 = data.slice(index);

      //   console.log(data2);


      //   this.safeHtml = this.sanitizer.bypassSecurityTrustHtml(data2);
      //   console.log(this.safeHtml);


      // });

    });
  }

  ngOnInit(): void {
  }

}
